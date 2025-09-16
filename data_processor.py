import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import sqlite3
from pathlib import Path
import psutil
import gc
from collections import defaultdict
import hashlib
from datetime import datetime, timedelta
import pickle
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompetitionDataProcessor:
    def __init__(self, source_dir="D:/images", cache_dir="./cache"):
        self.source_dir = Path(source_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Database for tracking processed files
        self.db_path = self.cache_dir / "processed_files.db"
        self.init_database()
        
        # Model configurations
        self.clip_model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Processing queues and locks
        self.processing_queue = Queue()
        self.processing_lock = threading.Lock()
        self.stop_processing = False
        
        # Performance monitoring
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'processing_speed': 0,
            'memory_usage': 0,
            'last_update': None
        }
        
        # FAISS index configuration for large datasets
        self.dimension = 512  # CLIP ViT-B/32 dimension
        self.faiss_index = None
        self.image_paths = []
        self.metadata = {}
        
        # Batch processing
        self.batch_size = 128
        self.max_workers = 8
        
    def init_database(self):
        """Initialize SQLite database for tracking processed files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                file_size INTEGER,
                status TEXT DEFAULT 'pending',
                processed_at TIMESTAMP,
                error_message TEXT,
                thumbnail_path TEXT,
                metadata_path TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_files INTEGER,
                processed_files INTEGER,
                failed_files INTEGER,
                processing_speed REAL,
                memory_usage REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_models(self):
        """Load CLIP model with memory optimization"""
        try:
            logger.info("Loading CLIP model...")
            self.clip_model = SentenceTransformer('clip-ViT-B-32')
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def scan_source_directory(self):
        """Scan source directory for new files"""
        try:
            logger.info(f"Scanning source directory: {self.source_dir}")
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            new_files = []
            
            for file_path in self.source_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    new_files.append(file_path)
            
            logger.info(f"Found {len(new_files)} image files")
            
            # Check database for new files
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for file_path in new_files:
                file_hash = self.calculate_file_hash(file_path)
                file_size = file_path.stat().st_size
                
                cursor.execute('''
                    INSERT OR IGNORE INTO processed_files 
                    (file_path, file_hash, file_size, status) 
                    VALUES (?, ?, ?, 'pending')
                ''', (str(file_path), file_hash, file_size))
            
            conn.commit()
            conn.close()
            
            logger.info("File scanning completed")
            return len(new_files)
            
        except Exception as e:
            logger.error(f"Error scanning directory: {e}")
            return 0
    
    def create_thumbnail(self, image_path, thumbnail_path, size=(150, 150)):
        """Create optimized thumbnail"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Resize with high quality
            thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
            
            # Optimize JPEG quality
            cv2.imwrite(str(thumbnail_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return True
            
        except Exception as e:
            logger.error(f"Error creating thumbnail for {image_path}: {e}")
            return False
    
    def extract_features(self, image_path):
        """Extract CLIP features from image"""
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            # Convert to PIL Image for CLIP
            from PIL import Image
            pil_image = Image.fromarray(image)
            
            # Extract features
            features = self.clip_model.encode([pil_image])[0]
            return features.astype('float32')
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def process_file_batch(self, file_batch):
        """Process a batch of files"""
        results = []
        
        for file_path in file_batch:
            try:
                # Create thumbnail
                thumbnail_path = self.cache_dir / "thumbnails" / f"{hashlib.md5(str(file_path).encode()).hexdigest()}.jpg"
                thumbnail_path.parent.mkdir(exist_ok=True)
                
                if not self.create_thumbnail(file_path, thumbnail_path):
                    continue
                
                # Extract features
                features = self.extract_features(file_path)
                if features is None:
                    continue
                
                # Extract metadata
                metadata = {
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'created_time': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    'thumbnail_path': str(thumbnail_path),
                    'features_shape': features.shape
                }
                
                results.append({
                    'file_path': str(file_path),
                    'features': features,
                    'metadata': metadata,
                    'thumbnail_path': str(thumbnail_path)
                })
                
                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE processed_files 
                    SET status = 'processed', processed_at = CURRENT_TIMESTAMP,
                        thumbnail_path = ?, metadata_path = ?
                    WHERE file_path = ?
                ''', (str(thumbnail_path), str(metadata), str(file_path)))
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
                # Mark as failed in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE processed_files 
                    SET status = 'failed', error_message = ?
                    WHERE file_path = ?
                ''', (str(e), str(file_path)))
                conn.commit()
                conn.close()
        
        return results
    
    def build_faiss_index(self, features_list):
        """Build FAISS index for large-scale similarity search"""
        try:
            if not features_list:
                logger.warning("No features to build index")
                return None
            
            # Convert to numpy array
            features_array = np.vstack(features_list).astype('float32')
            
            # Normalize features
            faiss.normalize_L2(features_array)
            
            # Choose index type based on dataset size
            if len(features_array) < 1000000:  # Less than 1M vectors
                # Use IndexFlatIP for smaller datasets (faster for small datasets)
                index = faiss.IndexFlatIP(self.dimension)
            else:
                # Use IndexIVFFlat for larger datasets (memory efficient)
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = min(4096, len(features_array) // 30)  # Number of clusters
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train the index
                logger.info("Training FAISS index...")
                index.train(features_array)
            
            # Add vectors to index
            logger.info(f"Adding {len(features_array)} vectors to FAISS index...")
            index.add(features_array)
            
            # Save index
            faiss.write_index(index, str(self.cache_dir / "faiss_index.bin"))
            logger.info(f"FAISS index saved with {index.ntotal} vectors")
            
            return index
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return None
    
    def save_metadata(self, metadata_list):
        """Save metadata to JSON file"""
        try:
            metadata_dict = {}
            for i, metadata in enumerate(metadata_list):
                metadata_dict[str(i)] = metadata
            
            with open(self.cache_dir / "competition_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Metadata saved for {len(metadata_list)} files")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def save_image_paths(self, image_paths):
        """Save image paths to JSON file"""
        try:
            with open(self.cache_dir / "image_path.json", 'w', encoding='utf-8') as f:
                json.dump(image_paths, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Image paths saved for {len(image_paths)} files")
            
        except Exception as e:
            logger.error(f"Error saving image paths: {e}")
    
    def process_all_files(self):
        """Process all pending files"""
        try:
            logger.info("Starting file processing...")
            
            # Get pending files from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT file_path FROM processed_files 
                WHERE status = 'pending'
                ORDER BY file_size ASC
            ''')
            pending_files = [Path(row[0]) for row in cursor.fetchall()]
            conn.close()
            
            if not pending_files:
                logger.info("No pending files to process")
                return
            
            logger.info(f"Processing {len(pending_files)} files...")
            
            # Process files in batches
            all_features = []
            all_metadata = []
            all_image_paths = []
            
            for i in range(0, len(pending_files), self.batch_size):
                batch = pending_files[i:i + self.batch_size]
                
                # Process batch with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    batch_results = list(executor.map(self.process_file_batch, [batch]))
                
                # Collect results
                for batch_result in batch_results:
                    for result in batch_result:
                        all_features.append(result['features'])
                        all_metadata.append(result['metadata'])
                        all_image_paths.append(result['file_path'])
                
                # Update progress
                processed = min(i + self.batch_size, len(pending_files))
                logger.info(f"Processed {processed}/{len(pending_files)} files")
                
                # Memory management
                if len(all_features) % (self.batch_size * 10) == 0:
                    gc.collect()
            
            # Build FAISS index
            if all_features:
                logger.info("Building FAISS index...")
                self.faiss_index = self.build_faiss_index(all_features)
                
                # Save metadata and paths
                self.save_metadata(all_metadata)
                self.save_image_paths(all_image_paths)
                
                # Create flag for app.py to reload data
                with open('new_data_flag.txt', 'w') as f:
                    f.write(str(datetime.now()))
                
                logger.info("Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in file processing: {e}")
    
    def start_monitoring(self):
        """Start file system monitoring"""
        class FileHandler(FileSystemEventHandler):
            def __init__(self, processor):
                self.processor = processor
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    logger.info(f"New file detected: {event.src_path}")
                    # Add to processing queue
                    self.processor.processing_queue.put(event.src_path)
            
            def on_moved(self, event):
                if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    logger.info(f"File moved: {event.src_path}")
                    self.processor.processing_queue.put(event.src_path)
        
        event_handler = FileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.source_dir), recursive=True)
        observer.start()
        
        logger.info(f"Started monitoring directory: {self.source_dir}")
        return observer
    
    def run_continuous_processing(self):
        """Run continuous processing loop"""
        try:
            # Load models
            self.load_models()
            
            # Initial scan
            self.scan_source_directory()
            
            # Process existing files
            self.process_all_files()
            
            # Start monitoring
            observer = self.start_monitoring()
            
            # Continuous processing loop
            while not self.stop_processing:
                try:
                    # Process new files from queue
                    while not self.processing_queue.empty():
                        file_path = self.processing_queue.get_nowait()
                        self.process_file_batch([Path(file_path)])
                    
                    # Periodic full scan (every 5 minutes)
                    if time.time() % 300 < 1:
                        self.scan_source_directory()
                        self.process_all_files()
                    
                    # Update stats
                    self.update_stats()
                    
                    # Sleep
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error in continuous processing: {e}")
                    time.sleep(30)
            
            observer.stop()
            observer.join()
            
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
    
    def update_stats(self):
        """Update processing statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM processed_files')
            total_files = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM processed_files WHERE status = "processed"')
            processed_files = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM processed_files WHERE status = "failed"')
            failed_files = cursor.fetchone()[0]
            
            conn.close()
            
            self.stats.update({
                'total_files': total_files,
                'processed_files': processed_files,
                'failed_files': failed_files,
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
                'last_update': datetime.now()
            })
            
            # Save stats to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO processing_stats 
                (total_files, processed_files, failed_files, processing_speed, memory_usage)
                VALUES (?, ?, ?, ?, ?)
            ''', (total_files, processed_files, failed_files, self.stats['processing_speed'], self.stats['memory_usage']))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def get_stats(self):
        """Get current processing statistics"""
        return self.stats.copy()

def main():
    """Main function to run the data processor"""
    processor = CompetitionDataProcessor()
    
    try:
        logger.info("Starting Competition Data Processor...")
        processor.run_continuous_processing()
    except KeyboardInterrupt:
        logger.info("Stopping processor...")
        processor.stop_processing = True
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()



