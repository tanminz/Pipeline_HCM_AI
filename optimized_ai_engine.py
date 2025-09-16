#!/usr/bin/env python3
"""
Optimized AI Search Engine for HCMC AI Challenge V
Following the detailed pipeline: Preprocessing -> Indexing -> Retrieval
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from collections import defaultdict
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI/ML imports with error handling
try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel
    from sentence_transformers import SentenceTransformer
    import faiss
    from PIL import Image
    import cv2
    from sklearn.metrics.pairwise import cosine_similarity
    import hashlib
except ImportError as e:
    logger.error(f"Critical AI library missing: {e}")
    raise

class OptimizedAISearchEngine:
    def __init__(self, metadata_file="real_image_metadata.json"):
        self.metadata_file = metadata_file
        self.image_metadata = {}
        self.clip_model = None
        self.clip_processor = None
        self.faiss_index = None
        self.clip_features = None
        self.feature_dim = 512  # CLIP ViT-B/32 dimension
        self.index_file = "faiss_index.bin"
        self.features_file = "clip_features.npy"
        self.valid_image_ids = []
        
        # Performance tracking
        self.search_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load metadata and initialize
        self.load_metadata()
        self.initialize_models()
        self.load_or_build_index()
    
    def load_metadata(self):
        """Load image metadata from JSON file"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.image_metadata = json.load(f)
            logger.info(f"✅ Loaded metadata for {len(self.image_metadata)} images")
        except Exception as e:
            logger.error(f"❌ Error loading metadata: {e}")
            self.image_metadata = {}
    
    def initialize_models(self):
        """Initialize CLIP model and processor"""
        try:
            logger.info("🚀 Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Set to evaluation mode
            self.clip_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                logger.info("✅ CLIP model loaded on GPU")
            else:
                logger.info("✅ CLIP model loaded on CPU")
                
        except Exception as e:
            logger.error(f"❌ Error loading CLIP model: {e}")
            raise
    
    def extract_clip_features(self, image_path: str) -> np.ndarray:
        """Extract CLIP features from an image with error handling"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                features = image_features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting CLIP features from {image_path}: {e}")
            return np.random.rand(self.feature_dim)  # Fallback
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features using CLIP"""
        try:
            # Process text
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                features = text_features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return np.random.rand(self.feature_dim)  # Fallback
    
    def load_or_build_index(self):
        """Load existing FAISS index or build new one"""
        try:
            # Try to load existing index
            if os.path.exists(self.index_file) and os.path.exists(self.features_file):
                logger.info("📂 Loading existing FAISS index...")
                self.faiss_index = faiss.read_index(self.index_file)
                self.clip_features = np.load(self.features_file)
                
                # Load valid image IDs
                with open("valid_image_ids.pkl", "rb") as f:
                    self.valid_image_ids = pickle.load(f)
                
                logger.info(f"✅ Loaded FAISS index with {len(self.valid_image_ids)} images")
                return
            
            # Build new index
            logger.info("🔨 Building new FAISS index...")
            self.build_search_index()
            
        except Exception as e:
            logger.error(f"Error loading/building index: {e}")
            self.build_search_index()
    
    def build_search_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            logger.info("🔨 Building FAISS search index...")
            
            # Extract features for all images
            features_list = []
            valid_images = []
            
            total_images = len(self.image_metadata)
            for i, (image_id_str, data) in enumerate(self.image_metadata.items()):
                if i % 1000 == 0:
                    logger.info(f"Processing {i}/{total_images} images...")
                
                image_path = data['web_path']
                if os.path.exists(image_path):
                    features = self.extract_clip_features(image_path)
                    features_list.append(features)
                    valid_images.append(image_id_str)
            
            if not features_list:
                logger.warning("No valid images found for indexing")
                return
            
            # Convert to numpy array
            self.clip_features = np.array(features_list, dtype=np.float32)
            
            # Normalize features for cosine similarity
            faiss.normalize_L2(self.clip_features)
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)  # Inner product for cosine similarity
            self.faiss_index.add(self.clip_features)
            
            # Store valid image IDs
            self.valid_image_ids = valid_images
            
            # Save index and features
            faiss.write_index(self.faiss_index, self.index_file)
            np.save(self.features_file, self.clip_features)
            
            with open("valid_image_ids.pkl", "wb") as f:
                pickle.dump(self.valid_image_ids, f)
            
            logger.info(f"✅ FAISS index built and saved with {len(valid_images)} images")
            
        except Exception as e:
            logger.error(f"Error building search index: {e}")
    
    def search_by_text(self, query: str, k: int = 50) -> List[Dict]:
        """Search images by text query using CLIP + FAISS"""
        start_time = time.time()
        
        try:
            if self.faiss_index is None or self.clip_model is None:
                logger.warning("AI models not available, using fallback search")
                return self.fallback_search(query, k)
            
            # Extract query features
            query_features = self.extract_text_features(query)
            query_features = query_features.reshape(1, -1).astype(np.float32)
            
            # Normalize query features
            faiss.normalize_L2(query_features)
            
            # Search using FAISS
            similarities, indices = self.faiss_index.search(query_features, k)
            
            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.valid_image_ids):
                    image_id_str = self.valid_image_ids[idx]
                    data = self.image_metadata[image_id_str]
                    
                    results.append({
                        'id': data['id'],
                        'path': data['web_path'],
                        'filename': image_id_str,
                        'similarity': float(similarity),
                        'rank': i + 1,
                        'source_zip': data.get('source_zip', 'Unknown')
                    })
            
            # Track performance
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            
            logger.info(f"🔍 Search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return self.fallback_search(query, k)
    
    def advanced_search(self, query: str, filters: Dict = None, k: int = 50) -> List[Dict]:
        """Advanced search with multiple strategies and filtering"""
        try:
            # Basic text search
            results = self.search_by_text(query, k * 2)  # Get more results for filtering
            
            # Apply filters if provided
            if filters:
                results = self.apply_filters(results, filters)
            
            # Re-rank based on additional criteria
            results = self.rerank_results(results, query)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return self.fallback_search(query, k)
    
    def apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            # Apply source filter
            if 'source' in filters:
                source_zip = result.get('source_zip', '')
                if filters['source'] not in source_zip:
                    continue
            
            # Apply size filter
            if 'min_size' in filters:
                size = self.image_metadata[result['filename']].get('size', 0)
                if size < filters['min_size']:
                    continue
            
            # Apply similarity threshold
            if 'min_similarity' in filters:
                if result['similarity'] < filters['min_similarity']:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Re-rank results based on additional criteria"""
        try:
            query_words = query.lower().split()
            
            for result in results:
                filename = result['filename'].lower()
                boost = 0.0
                
                # Boost for exact word matches in filename
                for word in query_words:
                    if word in filename:
                        boost += 0.1
                
                # Boost for source relevance
                source_zip = result.get('source_zip', '')
                if any(word in source_zip.lower() for word in query_words):
                    boost += 0.05
                
                # Apply boost
                result['similarity'] = min(1.0, result['similarity'] + boost)
            
            # Sort by updated similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results
    
    def fallback_search(self, query: str, k: int) -> List[Dict]:
        """Fallback search when AI models are not available"""
        try:
            query_lower = query.lower()
            results = []
            
            for image_id_str, data in self.image_metadata.items():
                filename_lower = image_id_str.lower()
                
                # Check if query words appear in filename
                query_words = query_lower.split()
                matches = sum(1 for word in query_words if word in filename_lower)
                
                if matches > 0:
                    similarity = matches / len(query_words)
                    results.append({
                        'id': data['id'],
                        'path': data['web_path'],
                        'filename': image_id_str,
                        'similarity': similarity,
                        'rank': len(results) + 1,
                        'source_zip': data.get('source_zip', 'Unknown')
                    })
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
    
    def get_search_statistics(self) -> Dict:
        """Get search engine statistics"""
        avg_search_time = np.mean(self.search_times) if self.search_times else 0
        
        return {
            'total_images': len(self.image_metadata),
            'indexed_images': len(self.valid_image_ids),
            'clip_model_loaded': self.clip_model is not None,
            'faiss_index_built': self.faiss_index is not None,
            'feature_dimension': self.feature_dim,
            'average_search_time': avg_search_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'gpu_available': torch.cuda.is_available()
        }
    
    def test_search(self, test_queries: List[str] = None):
        """Test the search engine with sample queries"""
        if test_queries is None:
            test_queries = [
                "person",
                "car",
                "building", 
                "nature",
                "indoor scene",
                "outdoor",
                "people",
                "vehicle"
            ]
        
        print("🧪 Testing AI Search Engine...")
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = self.search_by_text(query, k=5)
            
            print(f"Found {len(results)} results:")
            for result in results[:3]:  # Show top 3
                print(f"  - {result['filename']} (similarity: {result['similarity']:.3f})")
        
        # Show statistics
        stats = self.get_search_statistics()
        print(f"\n📊 Search Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

def main():
    """Main function to test the optimized AI search engine"""
    print("🚀 Initializing Optimized AI Search Engine...")
    
    try:
        engine = OptimizedAISearchEngine()
        engine.test_search()
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Search Engine: {e}")
        print("❌ AI Search Engine initialization failed!")

if __name__ == "__main__":
    main()
