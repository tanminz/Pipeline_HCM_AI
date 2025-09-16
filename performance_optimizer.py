#!/usr/bin/env python3
"""
Performance Optimizer for AI Challenge V - Event Retrieval System
Optimized for 100GB+ dataset with 5+ concurrent users
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import time
import logging
from typing import List, Dict, Any
import gc
import psutil
import threading
from queue import Queue
import faiss
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import cv2
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedImageDataset(Dataset):
    """Optimized dataset for large-scale image processing"""
    
    def __init__(self, image_paths: List[str], batch_size: int = 32):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.cache = {}
        self.cache_size = 1000
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # Load and preprocess image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize for memory efficiency
            image = image.resize((224, 224), Image.LANCZOS)
            
            # Cache if not full
            if len(self.cache) < self.cache_size:
                self.cache[idx] = image
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

class MemoryOptimizedProcessor:
    """Memory-optimized processor for large datasets"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.current_memory = 0
        self.lock = threading.Lock()
        
    def check_memory(self):
        """Check current memory usage"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024**3
        return memory_gb
    
    def cleanup_memory(self):
        """Force garbage collection"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def process_in_batches(self, data: List[Any], batch_size: int, processor_func):
        """Process data in memory-efficient batches"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Check memory before processing
            if self.check_memory() > self.max_memory_gb:
                logger.warning("Memory limit reached, cleaning up...")
                self.cleanup_memory()
            
            # Process batch
            batch_results = processor_func(batch)
            results.extend(batch_results)
            
            # Clean up after each batch
            del batch
            self.cleanup_memory()
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        
        return results

class OptimizedFeatureExtractor:
    """Optimized feature extraction with multiple models"""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.models = {}
        self.transform = self._get_transform()
        self.memory_processor = MemoryOptimizedProcessor()
        
        logger.info(f"Using device: {self.device}")
    
    def _get_transform(self):
        """Get optimized image transform"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_models(self):
        """Load models with memory optimization"""
        logger.info("Loading models...")
        
        # Load CLIP model
        self.models['clip'] = SentenceTransformer('clip-ViT-B-32')
        
        # Load InceptionResNetV2 (lighter than full InceptionV3)
        from torchvision import models
        self.models['inception'] = models.inception_v3(pretrained=True, aux_logits=False)
        self.models['inception'].eval()
        self.models['inception'].to(self.device)
        
        # Load Faster R-CNN (only if needed)
        if torch.cuda.is_available():
            self.models['faster_rcnn'] = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.models['faster_rcnn'].eval()
            self.models['faster_rcnn'].to(self.device)
        
        logger.info("Models loaded successfully")
    
    def extract_features_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Extract features from batch of images"""
        features_list = []
        
        for image in images:
            if image is None:
                features_list.append(None)
                continue
            
            try:
                # Preprocess
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                features = {}
                
                # CLIP features
                with torch.no_grad():
                    clip_features = self.models['clip'].encode([image])
                    features['clip'] = clip_features[0]
                
                # Inception features
                with torch.no_grad():
                    inception_features = self.models['inception'](image_tensor)
                    features['inception'] = inception_features.squeeze().cpu().numpy()
                
                # Object detection (optional, memory intensive)
                if 'faster_rcnn' in self.models:
                    with torch.no_grad():
                        detections = self.models['faster_rcnn'](image_tensor)
                        features['objects'] = self._process_detections(detections)
                
                features_list.append(features)
                
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                features_list.append(None)
        
        return features_list
    
    def _process_detections(self, detections):
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

class OptimizedIndexBuilder:
    """Optimized FAISS index builder for large datasets"""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.embeddings = []
        self.image_paths = []
        
    def build_index(self, features_list: List[Dict], image_paths: List[str], 
                   index_type: str = 'ivf'):
        """Build optimized FAISS index"""
        logger.info("Building optimized FAISS index...")
        
        # Combine features
        combined_features = []
        valid_paths = []
        
        for i, (features, path) in enumerate(zip(features_list, image_paths)):
            if features is not None:
                # Combine CLIP and Inception features
                combined_feature = np.concatenate([
                    features['clip'],
                    features['inception'][:512]  # Use first 512 dimensions
                ])
                combined_features.append(combined_feature)
                valid_paths.append(path)
            
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(features_list)} features")
        
        combined_features = np.array(combined_features, dtype='float32')
        
        # Build optimized index based on size
        if len(combined_features) > 1000000:  # 1M+ vectors
            # Use IVF index for very large datasets
            nlist = min(4096, len(combined_features) // 100)
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(self.dimension), 
                self.dimension, 
                nlist
            )
            # Train the index
            self.index.train(combined_features)
        else:
            # Use simple index for smaller datasets
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors
        self.index.add(combined_features)
        
        self.embeddings = combined_features
        self.image_paths = valid_paths
        
        logger.info(f"FAISS index built with {len(combined_features)} vectors")
        return self.index
    
    def save_index(self, filepath: str):
        """Save index and metadata"""
        faiss.write_index(self.index, filepath)
        
        metadata = {
            'embeddings_shape': self.embeddings.shape,
            'image_paths': self.image_paths
        }
        
        with open(filepath.replace('.bin', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {filepath}")

class SceneClusterer:
    """Optimized scene clustering for large datasets"""
    
    def __init__(self, n_clusters: int = 1000):
        self.n_clusters = n_clusters
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42
        )
        
    def cluster_scenes(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        """Cluster images into scenes using MiniBatchKMeans"""
        logger.info("Clustering scenes...")
        
        # Use CLIP embeddings for clustering
        clip_embeddings = embeddings[:, :512]  # First 512 dimensions are CLIP
        
        # Normalize
        clip_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
        
        # Cluster
        cluster_labels = self.clusterer.fit_predict(clip_embeddings)
        
        # Group by cluster
        scene_groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in scene_groups:
                scene_groups[label] = []
            scene_groups[label].append(i)
        
        logger.info(f"Found {len(scene_groups)} scene clusters")
        return scene_groups

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def log_metric(self, name: str, value: float):
        """Log a performance metric"""
        self.metrics[name] = value
    
    def get_system_info(self) -> Dict:
        """Get current system information"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def print_report(self):
        """Print performance report"""
        logger.info("=== Performance Report ===")
        for name, value in self.metrics.items():
            logger.info(f"{name}: {value}")
        
        system_info = self.get_system_info()
        logger.info("=== System Info ===")
        for name, value in system_info.items():
            logger.info(f"{name}: {value}")

def optimize_for_competition(image_paths: List[str], output_dir: str = 'optimized_data'):
    """Main optimization function for competition"""
    logger.info("Starting competition optimization...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    monitor = PerformanceMonitor()
    extractor = OptimizedFeatureExtractor()
    index_builder = OptimizedIndexBuilder()
    clusterer = SceneClusterer()
    
    # Load models
    extractor.load_models()
    monitor.log_metric('model_loading_time', time.time() - monitor.start_time)
    
    # Process images in batches
    batch_size = 64 if torch.cuda.is_available() else 32
    dataset = OptimizedImageDataset(image_paths, batch_size)
    
    # Extract features
    features_list = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_features = extractor.extract_features_batch(batch)
        features_list.extend(batch_features)
        
        # Monitor memory
        system_info = monitor.get_system_info()
        if system_info['memory_percent'] > 90:
            logger.warning("High memory usage, cleaning up...")
            extractor.memory_processor.cleanup_memory()
    
    monitor.log_metric('feature_extraction_time', time.time() - monitor.start_time)
    
    # Build index
    index = index_builder.build_index(features_list, image_paths)
    index_builder.save_index(os.path.join(output_dir, 'faiss_index.bin'))
    monitor.log_metric('index_building_time', time.time() - monitor.start_time)
    
    # Cluster scenes
    scene_groups = clusterer.cluster_scenes(index_builder.embeddings)
    with open(os.path.join(output_dir, 'scene_clusters.pkl'), 'wb') as f:
        pickle.dump(scene_groups, f)
    monitor.log_metric('clustering_time', time.time() - monitor.start_time)
    
    # Save optimized data
    optimized_data = {
        'image_paths': index_builder.image_paths,
        'embeddings_shape': index_builder.embeddings.shape,
        'scene_count': len(scene_groups),
        'total_images': len(index_builder.image_paths)
    }
    
    with open(os.path.join(output_dir, 'optimized_metadata.json'), 'w') as f:
        json.dump(optimized_data, f, indent=2)
    
    # Print final report
    monitor.print_report()
    
    logger.info("Optimization completed successfully!")
    return optimized_data

def create_optimized_config():
    """Create optimized configuration for production"""
    config = {
        'server': {
            'host': '0.0.0.0',
            'port': 5001,
            'threaded': True,
            'workers': min(4, mp.cpu_count()),
            'max_connections': 100
        },
        'cache': {
            'type': 'redis',  # Use Redis for distributed caching
            'host': 'localhost',
            'port': 6379,
            'timeout': 300,
            'max_entries': 10000
        },
        'models': {
            'clip_model': 'clip-ViT-B-32',
            'inception_model': 'inception_v3',
            'use_gpu': torch.cuda.is_available(),
            'batch_size': 64 if torch.cuda.is_available() else 32
        },
        'index': {
            'type': 'ivf',
            'nlist': 4096,
            'nprobe': 16
        },
        'performance': {
            'max_memory_gb': 8.0,
            'max_concurrent_users': 5,
            'request_timeout': 30,
            'enable_compression': True
        }
    }
    
    with open('optimized_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Optimized configuration created")

if __name__ == '__main__':
    # Example usage
    logger.info("Performance Optimizer for AI Challenge V")
    
    # Create optimized configuration
    create_optimized_config()
    
    # Example: optimize dataset (uncomment when ready)
    # image_paths = load_image_paths()  # Load your image paths
    # optimized_data = optimize_for_competition(image_paths)
    
    logger.info("Optimization setup completed!")


