import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, request, send_file, jsonify, redirect, url_for
from flask_caching import Cache
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache configuration for high performance
cache_config = {
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_THRESHOLD': 10000
}
cache = Cache(app, config=cache_config)

# Global variables for models and data
clip_model = None
faster_rcnn_model = None
inception_model = None
image_embeddings = None
text_embeddings = None
image_paths = None
metadata = None
faiss_index = None
scene_clusters = None
video_segments = None

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OptimizedImageRetrieval:
    def __init__(self):
        self.clip_model = None
        self.faster_rcnn_model = None
        self.inception_model = None
        self.image_embeddings = None
        self.text_embeddings = None
        self.faiss_index = None
        self.scene_clusters = None
        self.video_segments = None
        
    def load_models(self):
        """Load all models with optimization"""
        logger.info("Loading models...")
        
        # Load CLIP model
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        
        # Load Faster R-CNN for object detection
        self.faster_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.faster_rcnn_model.eval()
        
        # Load InceptionResNetV2 for feature extraction
        self.inception_model = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception_model.eval()
        
        logger.info("Models loaded successfully")
    
    def extract_features(self, image_path):
        """Extract features from image using multiple models"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            features = {}
            
            # CLIP features
            with torch.no_grad():
                clip_features = self.clip_model.encode([image])
                features['clip'] = clip_features[0]
            
            # InceptionResNetV2 features
            with torch.no_grad():
                inception_features = self.inception_model(image_tensor)
                features['inception'] = inception_features.squeeze().numpy()
            
            # Faster R-CNN object detection
            with torch.no_grad():
                detections = self.faster_rcnn_model(image_tensor)
                features['objects'] = self.process_detections(detections)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def process_detections(self, detections):
        """Process Faster R-CNN detections"""
        boxes = detections[0]['boxes'].cpu().numpy()
        scores = detections[0]['scores'].cpu().numpy()
        labels = detections[0]['labels'].cpu().numpy()
        
        # Filter high-confidence detections
        high_conf_mask = scores > 0.5
        return {
            'boxes': boxes[high_conf_mask],
            'scores': scores[high_conf_mask],
            'labels': labels[high_conf_mask]
        }
    
    def build_index(self, image_paths):
        """Build FAISS index for fast similarity search"""
        logger.info("Building FAISS index...")
        
        embeddings = []
        valid_paths = []
        
        for i, path in enumerate(image_paths):
            features = self.extract_features(path)
            if features is not None:
                # Combine CLIP and Inception features
                combined_features = np.concatenate([
                    features['clip'],
                    features['inception'][:512]  # Use first 512 dimensions
                ])
                embeddings.append(combined_features)
                valid_paths.append(path)
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(image_paths)} images")
        
        embeddings = np.array(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.image_embeddings = embeddings
        self.image_paths = valid_paths
        
        logger.info(f"FAISS index built with {len(embeddings)} images")
    
    def cluster_scenes(self, eps=0.3, min_samples=5):
        """Cluster images into scenes using DBSCAN"""
        logger.info("Clustering scenes...")
        
        # Use CLIP embeddings for scene clustering
        clip_embeddings = np.array([emb[:512] for emb in self.image_embeddings])
        
        # Normalize for cosine similarity
        clip_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(clip_embeddings)
        
        self.scene_clusters = cluster_labels
        
        # Group images by scene
        scene_groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in scene_groups:
                scene_groups[label] = []
            scene_groups[label].append(i)
        
        logger.info(f"Found {len(scene_groups)} scene clusters")
        return scene_groups
    
    def search_similar_images(self, query_image_path, k=20):
        """Search for similar images using FAISS"""
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return []
        
        # Combine features
        query_embedding = np.concatenate([
            query_features['clip'],
            query_features['inception'][:512]
        ]).reshape(1, -1).astype('float32')
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.image_paths):
                results.append({
                    'path': self.image_paths[idx],
                    'similarity': float(sim),
                    'rank': i + 1
                })
        
        return results
    
    def search_by_text(self, text_query, k=20):
        """Search images by text description using CLIP"""
        # Encode text query
        text_embedding = self.clip_model.encode([text_query])[0]
        
        # Search in CLIP embeddings
        clip_embeddings = np.array([emb[:512] for emb in self.image_embeddings])
        
        # Calculate similarities
        similarities = cosine_similarity([text_embedding], clip_embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'path': self.image_paths[idx],
                'similarity': float(similarities[idx]),
                'rank': i + 1
            })
        
        return results

# Initialize retrieval system
retrieval_system = OptimizedImageRetrieval()

def load_data():
    """Load all data with caching"""
    global image_paths, metadata, video_segments
    
    # Load image paths
    if os.path.exists('image_path.json'):
        with open('image_path.json', 'r') as f:
            image_paths = json.load(f)
    
    # Load metadata
    if os.path.exists('competition_metadata.json'):
        with open('competition_metadata.json', 'r') as f:
            metadata = json.load(f)
    
    # Load video segments
    if os.path.exists('video_segments.json'):
        with open('video_segments.json', 'r') as f:
            video_segments = json.load(f)
    
    logger.info(f"Loaded {len(image_paths) if image_paths else 0} images")

def initialize_system():
    """Initialize the entire system"""
    logger.info("Initializing system...")
    
    # Load data
    load_data()
    
    # Load models
    retrieval_system.load_models()
    
    # Build index if not exists
    if not os.path.exists('faiss_index.bin'):
        if image_paths:
            retrieval_system.build_index(image_paths)
            # Save index
            faiss.write_index(retrieval_system.faiss_index, 'faiss_index.bin')
            # Save embeddings
            with open('embeddings.pkl', 'wb') as f:
                pickle.dump({
                    'embeddings': retrieval_system.image_embeddings,
                    'paths': retrieval_system.image_paths
                }, f)
    else:
        # Load existing index
        retrieval_system.faiss_index = faiss.read_index('faiss_index.bin')
        with open('embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
            retrieval_system.image_embeddings = data['embeddings']
            retrieval_system.image_paths = data['paths']
    
    # Cluster scenes
    if not os.path.exists('scene_clusters.pkl'):
        scene_groups = retrieval_system.cluster_scenes()
        with open('scene_clusters.pkl', 'wb') as f:
            pickle.dump(scene_groups, f)
    else:
        with open('scene_clusters.pkl', 'rb') as f:
            scene_groups = pickle.load(f)
    
    logger.info("System initialized successfully")

# Routes
@app.route('/')
def home():
    """Home page with optimized template"""
    return render_template('home_optimized.html', data={'query': ''})

@app.route('/home')
def home_route():
    """Home route with pagination"""
    index = request.args.get('index', 0, type=int)
    imgpath = request.args.get('imgpath', '')
    
    # Get paginated data
    page_size = 20
    start_idx = index * page_size
    end_idx = start_idx + page_size
    
    if retrieval_system.image_paths:
        page_paths = retrieval_system.image_paths[start_idx:end_idx]
        total_pages = (len(retrieval_system.image_paths) + page_size - 1) // page_size
        
        page_data = []
        for i, path in enumerate(page_paths):
            page_data.append({
                'id': start_idx + i,
                'imgpath': path
            })
        
        data = {
            'pagefile': page_data,
            'num_page': total_pages,
            'query': ''
        }
    else:
        data = {'pagefile': [], 'num_page': 0, 'query': ''}
    
    return render_template('home_optimized.html', data=data)

@app.route('/textsearch')
@cache.memoize(timeout=300)
def text_search():
    """Text search with caching"""
    text_query = request.args.get('textquery', '')
    search_type = request.args.get('search_type', 'single')
    
    if not text_query:
        return redirect('/')
    
    # Search by text
    results = retrieval_system.search_by_text(text_query, k=50)
    
    # Format results
    page_data = []
    for result in results:
        page_data.append({
            'id': result['rank'],
            'imgpath': result['path'],
            'similarity': result['similarity']
        })
    
    data = {
        'pagefile': page_data,
        'num_page': 1,
        'query': text_query
    }
    
    return render_template('home_optimized.html', data=data)

@app.route('/imgsearch')
@cache.memoize(timeout=300)
def image_search():
    """Image similarity search with caching"""
    imgid = request.args.get('imgid', type=int)
    
    if imgid is None or imgid >= len(retrieval_system.image_paths):
        return redirect('/')
    
    query_path = retrieval_system.image_paths[imgid]
    results = retrieval_system.search_similar_images(query_path, k=50)
    
    # Format results
    page_data = []
    for result in results:
        page_data.append({
            'id': result['rank'],
            'imgpath': result['path'],
            'similarity': result['similarity']
        })
    
    data = {
        'pagefile': page_data,
        'num_page': 1,
        'query': f'Similar to image {imgid}'
    }
    
    return render_template('home_optimized.html', data=data)

@app.route('/get_img')
def get_image():
    """Serve images with caching"""
    fpath = request.args.get('fpath', '')
    
    if not fpath or not os.path.exists(fpath):
        return send_file('static/images/404.jpg')
    
    return send_file(fpath)

@app.route('/showsegment')
def show_segment():
    """Show video segment for image"""
    imgid = request.args.get('imgid', type=int)
    
    if imgid is None or imgid >= len(retrieval_system.image_paths):
        return redirect('/')
    
    image_path = retrieval_system.image_paths[imgid]
    
    # Find video segment for this image
    segment_info = find_video_segment(image_path)
    
    return render_template('segment.html', 
                         image_path=image_path,
                         segment_info=segment_info)

@app.route('/api/search')
def api_search():
    """API endpoint for search"""
    query = request.args.get('q', '')
    search_type = request.args.get('type', 'text')
    
    if search_type == 'text':
        results = retrieval_system.search_by_text(query, k=20)
    else:
        results = retrieval_system.search_similar_images(query, k=20)
    
    return jsonify({
        'results': results,
        'query': query,
        'count': len(results)
    })

@app.route('/api/scene/<int:scene_id>')
def api_scene(scene_id):
    """API endpoint for scene information"""
    with open('scene_clusters.pkl', 'rb') as f:
        scene_groups = pickle.load(f)
    
    if scene_id not in scene_groups:
        return jsonify({'error': 'Scene not found'}), 404
    
    scene_images = []
    for idx in scene_groups[scene_id]:
        scene_images.append({
            'id': idx,
            'path': retrieval_system.image_paths[idx]
        })
    
    return jsonify({
        'scene_id': scene_id,
        'images': scene_images,
        'count': len(scene_images)
    })

def find_video_segment(image_path):
    """Find video segment information for image"""
    # Extract video info from image path
    # This is a placeholder - implement based on your data structure
    return {
        'video_path': 'path/to/video.mp4',
        'start_time': 0,
        'end_time': 10,
        'frame_number': 0
    }

# Performance monitoring
@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    return jsonify({
        'total_images': len(retrieval_system.image_paths) if retrieval_system.image_paths else 0,
        'index_size': retrieval_system.faiss_index.ntotal if retrieval_system.faiss_index else 0,
        'models_loaded': all([
            retrieval_system.clip_model,
            retrieval_system.faster_rcnn_model,
            retrieval_system.inception_model
        ])
    })

if __name__ == '__main__':
    # Initialize system in background thread
    init_thread = threading.Thread(target=initialize_system)
    init_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


