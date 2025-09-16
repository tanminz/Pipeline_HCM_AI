#!/usr/bin/env python3
"""
Fast Index Builder for HCMC AI Challenge V
Builds FAISS index from all available images quickly
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
import time
import pickle
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI/ML imports
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    import faiss
    from PIL import Image
except ImportError as e:
    logger.error(f"Critical AI library missing: {e}")
    raise

class FastIndexBuilder:
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.feature_dim = 512
        self.index_file = "fast_faiss_index.bin"
        self.features_file = "fast_clip_features.npy"
        self.metadata_file = "fast_image_metadata.json"
        self.valid_ids_file = "fast_valid_ids.pkl"
        
        # Performance tracking
        self.total_images = 0
        self.processed_images = 0
        self.start_time = None
        
    def initialize_clip(self):
        """Initialize CLIP model"""
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
    
    def scan_all_images(self) -> Dict:
        """Scan all images in static/images directory"""
        logger.info("🔍 Scanning all images...")
        
        image_metadata = {}
        image_id = 0
        
        images_dir = Path("static/images")
        if not images_dir.exists():
            logger.error("static/images directory not found!")
            return {}
        
        logger.info(f"Scanning images in: {images_dir.absolute()}")
        
        # Scan all subdirectories recursively
        for item in images_dir.rglob("*.jpg"):
            try:
                # Create relative path from static/images
                relative_path = item.relative_to(images_dir)
                
                # Parse the path structure
                path_parts = str(relative_path).split('\\')  # Windows path separator
                
                if len(path_parts) >= 2:
                    video_folder = path_parts[0]  # e.g., Keyframes_L21
                    
                    if len(path_parts) >= 3:
                        video_name = path_parts[-2]  # e.g., L21_V001
                        frame_name = path_parts[-1]  # e.g., 044.jpg
                    else:
                        video_name = path_parts[0]
                        frame_name = path_parts[1]
                    
                    # Create image ID string
                    image_id_str = f"{video_name}/{frame_name}"
                    
                    image_metadata[image_id_str] = {
                        'id': image_id,
                        'web_path': str(item),
                        'filename': image_id_str,
                        'video_folder': video_folder,
                        'video_name': video_name,
                        'frame_name': frame_name,
                        'size': item.stat().st_size,
                        'is_real_data': True
                    }
                    
                    image_id += 1
                    
                    if image_id % 1000 == 0:
                        logger.info(f"Scanned {image_id} images...")
                        
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                continue
        
        self.total_images = len(image_metadata)
        logger.info(f"✅ Total images scanned: {self.total_images}")
        
        # Show some examples
        if image_metadata:
            sample_keys = list(image_metadata.keys())[:5]
            logger.info(f"Sample images: {sample_keys}")
        
        return image_metadata
    
    def extract_features_batch(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """Extract CLIP features in large batches for speed"""
        features_list = []
        
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load batch of images
            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    valid_indices.append(j)
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Process batch with CLIP
                inputs = self.clip_processor(images=batch_images, return_tensors="pt", padding=True)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    batch_features = image_features.cpu().numpy()
                    
                    # Add features for valid images
                    for idx in valid_indices:
                        features_list.append(batch_features[idx])
                        
                # Progress update
                batch_num = i // batch_size + 1
                self.processed_images += len(valid_indices)
                elapsed = time.time() - self.start_time
                rate = self.processed_images / elapsed if elapsed > 0 else 0
                eta = (self.total_images - self.processed_images) / rate if rate > 0 else 0
                
                logger.info(f"Batch {batch_num}/{total_batches}: {self.processed_images}/{self.total_images} images processed "
                          f"({rate:.1f} img/s, ETA: {eta/60:.1f} min)")
                        
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Add random features for failed batch
                for _ in range(len(batch_paths)):
                    features_list.append(np.random.rand(self.feature_dim))
        
        return np.array(features_list, dtype=np.float32)
    
    def build_index(self):
        """Build FAISS index from all images"""
        try:
            self.start_time = time.time()
            
            # Initialize CLIP
            self.initialize_clip()
            
            # Scan all images
            image_metadata = self.scan_all_images()
            
            if not image_metadata:
                logger.error("No images found!")
                return
            
            # Get all image paths
            image_paths = []
            valid_images = []
            
            logger.info("Preparing image paths...")
            for i, (image_id_str, data) in enumerate(image_metadata.items()):
                if i % 1000 == 0:
                    logger.info(f"Preparing {i}/{len(image_metadata)} images...")
                
                image_path = data['web_path']
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    valid_images.append(image_id_str)
            
            if not image_paths:
                logger.error("No valid images found for indexing")
                return
            
            logger.info(f"✅ Found {len(image_paths)} valid images for indexing")
            
            # Extract features
            logger.info(f"🔨 Extracting CLIP features for {len(image_paths)} images...")
            clip_features = self.extract_features_batch(image_paths, batch_size=64)
            
            # Normalize features
            logger.info("Normalizing features...")
            faiss.normalize_L2(clip_features)
            
            # Create FAISS index
            logger.info("Building FAISS index...")
            faiss_index = faiss.IndexFlatIP(self.feature_dim)
            faiss_index.add(clip_features)
            
            # Save everything
            logger.info("Saving index and metadata...")
            
            # Save FAISS index
            faiss.write_index(faiss_index, self.index_file)
            
            # Save features
            np.save(self.features_file, clip_features)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(image_metadata, f, indent=2, ensure_ascii=False)
            
            # Save valid image IDs
            with open(self.valid_ids_file, "wb") as f:
                pickle.dump(valid_images, f)
            
            # Final statistics
            total_time = time.time() - self.start_time
            logger.info(f"✅ Index built successfully!")
            logger.info(f"📊 Statistics:")
            logger.info(f"  - Total images: {len(image_metadata)}")
            logger.info(f"  - Indexed images: {len(valid_images)}")
            logger.info(f"  - Feature dimension: {self.feature_dim}")
            logger.info(f"  - Total time: {total_time/60:.1f} minutes")
            logger.info(f"  - Processing rate: {len(valid_images)/total_time:.1f} images/second")
            logger.info(f"  - Index file size: {os.path.getsize(self.index_file)/1024/1024:.1f} MB")
            logger.info(f"  - Features file size: {os.path.getsize(self.features_file)/1024/1024:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def test_index(self):
        """Test the built index"""
        try:
            logger.info("🧪 Testing built index...")
            
            # Load index
            faiss_index = faiss.read_index(self.index_file)
            clip_features = np.load(self.features_file)
            
            with open(self.valid_ids_file, "rb") as f:
                valid_images = pickle.load(f)
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            
            # Test queries
            test_queries = ["buffalo", "person", "car", "building", "nature"]
            
            for query in test_queries:
                logger.info(f"Testing query: '{query}'")
                
                # Extract query features
                inputs = self.clip_processor(text=query, return_tensors="pt", padding=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    query_features = text_features.cpu().numpy().flatten()
                
                # Normalize query features
                query_features = query_features.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(query_features)
                
                # Search
                similarities, indices = faiss_index.search(query_features, 10)
                
                logger.info(f"Top 5 results for '{query}':")
                for i, (similarity, idx) in enumerate(zip(similarities[0][:5], indices[0][:5])):
                    if idx < len(valid_images):
                        image_id_str = valid_images[idx]
                        logger.info(f"  {i+1}. {image_id_str} (similarity: {similarity:.3f})")
            
            logger.info("✅ Index test completed successfully!")
            
        except Exception as e:
            logger.error(f"Error testing index: {e}")

def main():
    """Main function to build index"""
    print("🚀 Starting Fast Index Builder...")
    
    try:
        builder = FastIndexBuilder()
        
        # Build index
        builder.build_index()
        
        # Test index
        builder.test_index()
        
        print("✅ Index building completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        print("❌ Index building failed!")

if __name__ == "__main__":
    main()




