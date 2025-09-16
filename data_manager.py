#!/usr/bin/env python3
"""
Smart Data Manager for AI Challenge V
Handles incremental data loading from D:/images with real-time monitoring
"""

import os
import json
import time
import threading
import logging
from pathlib import Path
from typing import List, Dict, Set
import shutil
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataManager:
    """Smart data manager for handling large datasets"""
    
    def __init__(self, source_dir: str = "D:/images", cache_dir: str = "./cache"):
        self.source_dir = Path(source_dir)
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "data_index.db"
        self.processed_files: Set[str] = set()
        self.file_hashes: Dict[str, str] = {}
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "thumbnails").mkdir(exist_ok=True)
        (self.cache_dir / "features").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Start file watcher
        self.observer = Observer()
        self.file_handler = ImageFileHandler(self)
        self.observer.schedule(self.file_handler, str(self.source_dir), recursive=True)
        self.observer.start()
        
        logger.info(f"DataManager initialized - Source: {source_dir}, Cache: {cache_dir}")
    
    def init_database(self):
        """Initialize SQLite database for tracking files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                file_size INTEGER,
                created_time REAL,
                processed_time REAL,
                thumbnail_path TEXT,
                feature_path TEXT,
                metadata_path TEXT,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_files INTEGER,
                processed_files INTEGER,
                failed_files INTEGER,
                last_update REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def scan_existing_files(self):
        """Scan for existing files in source directory"""
        logger.info("Scanning existing files...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files_found = 0
        
        for file_path in self.source_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                self.add_file_to_database(file_path)
                files_found += 1
                
                if files_found % 1000 == 0:
                    logger.info(f"Scanned {files_found} files...")
        
        logger.info(f"Found {files_found} image files")
        self.update_stats()
    
    def add_file_to_database(self, file_path: Path):
        """Add file to database if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            file_hash = self.calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            created_time = file_path.stat().st_ctime
            
            cursor.execute('''
                INSERT OR IGNORE INTO files 
                (file_path, file_hash, file_size, created_time, status)
                VALUES (?, ?, ?, ?, 'pending')
            ''', (str(file_path), file_hash, file_size, created_time))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")
        finally:
            conn.close()
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def get_pending_files(self, limit: int = 100) -> List[Dict]:
        """Get list of pending files to process"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path, file_hash, file_size, created_time
            FROM files 
            WHERE status = 'pending'
            ORDER BY created_time ASC
            LIMIT ?
        ''', (limit,))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                'file_path': row[0],
                'file_hash': row[1],
                'file_size': row[2],
                'created_time': row[3]
            })
        
        conn.close()
        return files
    
    def mark_file_processed(self, file_path: str, thumbnail_path: str = None, 
                           feature_path: str = None, metadata_path: str = None):
        """Mark file as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE files 
            SET status = 'processed', processed_time = ?, 
                thumbnail_path = ?, feature_path = ?, metadata_path = ?
            WHERE file_path = ?
        ''', (time.time(), thumbnail_path, feature_path, metadata_path, file_path))
        
        conn.commit()
        conn.close()
    
    def mark_file_failed(self, file_path: str, error: str = None):
        """Mark file as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE files 
            SET status = 'failed', processed_time = ?
            WHERE file_path = ?
        ''', (time.time(), file_path))
        
        conn.commit()
        conn.close()
        
        if error:
            logger.error(f"Failed to process {file_path}: {error}")
    
    def update_stats(self):
        """Update processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM files')
        total_files = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM files WHERE status = "processed"')
        processed_files = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM files WHERE status = "failed"')
        failed_files = cursor.fetchone()[0]
        
        cursor.execute('''
            INSERT OR REPLACE INTO processing_stats 
            (id, total_files, processed_files, failed_files, last_update)
            VALUES (1, ?, ?, ?, ?)
        ''', (total_files, processed_files, failed_files, time.time()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stats: {processed_files}/{total_files} processed, {failed_files} failed")
    
    def get_stats(self) -> Dict:
        """Get current processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM processing_stats WHERE id = 1')
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return {
                'total_files': row[1],
                'processed_files': row[2],
                'failed_files': row[3],
                'last_update': row[4],
                'progress_percent': (row[2] / row[1] * 100) if row[1] > 0 else 0
            }
        return {'total_files': 0, 'processed_files': 0, 'failed_files': 0, 'progress_percent': 0}
    
    def get_processed_files(self, limit: int = 1000) -> List[str]:
        """Get list of processed file paths"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path FROM files 
            WHERE status = 'processed'
            ORDER BY processed_time DESC
            LIMIT ?
        ''', (limit,))
        
        files = [row[0] for row in cursor.fetchall()]
        conn.close()
        return files
    
    def cleanup_old_files(self, days: int = 30):
        """Clean up old processed files"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path FROM files 
            WHERE processed_time < ? AND status = 'processed'
        ''', (cutoff_time,))
        
        old_files = [row[0] for row in cursor.fetchall()]
        
        for file_path in old_files:
            try:
                os.remove(file_path)
                cursor.execute('DELETE FROM files WHERE file_path = ?', (file_path,))
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {len(old_files)} old files")

class ImageFileHandler(FileSystemEventHandler):
    """Handle file system events for new images"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
                logger.info(f"New image detected: {file_path}")
                # Wait a bit for file to be fully written
                threading.Timer(2.0, self.data_manager.add_file_to_database, args=[file_path]).start()
    
    def on_moved(self, event):
        if not event.is_directory:
            file_path = Path(event.dest_path)
            if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
                logger.info(f"Image moved: {file_path}")
                threading.Timer(2.0, self.data_manager.add_file_to_database, args=[file_path]).start()

class IncrementalProcessor:
    """Process files incrementally as they arrive"""
    
    def __init__(self, data_manager: DataManager, batch_size: int = 10):
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.running = False
        self.thread = None
    
    def start(self):
        """Start incremental processing"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Incremental processor started")
    
    def stop(self):
        """Stop incremental processing"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Incremental processor stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get pending files
                pending_files = self.data_manager.get_pending_files(self.batch_size)
                
                if pending_files:
                    logger.info(f"Processing batch of {len(pending_files)} files")
                    
                    # Process files in parallel
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [executor.submit(self._process_file, file_info) 
                                 for file_info in pending_files]
                        
                        for future in futures:
                            try:
                                future.result(timeout=30)  # 30 second timeout per file
                            except Exception as e:
                                logger.error(f"Error in file processing: {e}")
                    
                    # Update stats
                    self.data_manager.update_stats()
                else:
                    # No files to process, wait a bit
                    time.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(10)
    
    def _process_file(self, file_info: Dict):
        """Process a single file"""
        file_path = Path(file_info['file_path'])
        
        try:
            # Create thumbnail
            thumbnail_path = self._create_thumbnail(file_path)
            
            # Extract basic metadata
            metadata_path = self._extract_metadata(file_path)
            
            # Mark as processed
            self.data_manager.mark_file_processed(
                str(file_path),
                str(thumbnail_path) if thumbnail_path else None,
                metadata_path=str(metadata_path) if metadata_path else None
            )
            
        except Exception as e:
            self.data_manager.mark_file_failed(str(file_path), str(e))
    
    def _create_thumbnail(self, file_path: Path) -> Path:
        """Create thumbnail for image"""
        try:
            from PIL import Image
            
            thumbnail_path = self.data_manager.cache_dir / "thumbnails" / f"{file_path.stem}_thumb.jpg"
            
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail((200, 200), Image.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85)
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error creating thumbnail for {file_path}: {e}")
            return None
    
    def _extract_metadata(self, file_path: Path) -> Path:
        """Extract basic metadata from image"""
        try:
            from PIL import Image
            
            metadata_path = self.data_manager.cache_dir / "metadata" / f"{file_path.stem}_meta.json"
            
            with Image.open(file_path) as img:
                metadata = {
                    'filename': file_path.name,
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': file_path.stat().st_size,
                    'created_time': file_path.stat().st_ctime
                }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata_path
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {e}")
            return None

def create_data_manager():
    """Create and initialize data manager"""
    data_manager = DataManager()
    
    # Scan existing files
    data_manager.scan_existing_files()
    
    # Start incremental processor
    processor = IncrementalProcessor(data_manager)
    processor.start()
    
    return data_manager, processor

if __name__ == '__main__':
    # Test the data manager
    data_manager, processor = create_data_manager()
    
    try:
        while True:
            stats = data_manager.get_stats()
            print(f"Progress: {stats['progress_percent']:.1f}% "
                  f"({stats['processed_files']}/{stats['total_files']})")
            time.sleep(10)
    except KeyboardInterrupt:
        processor.stop()
        data_manager.observer.stop()
        data_manager.observer.join()


