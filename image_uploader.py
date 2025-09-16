#!/usr/bin/env python3
"""
Simple Image Uploader - Upload and optimize images for web display
"""

import os
import shutil
import logging
from pathlib import Path
from PIL import Image, ImageOps
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageUploader:
    def __init__(self):
        self.source_dir = Path("D:/images")
        self.web_dir = Path("static/images")
        self.thumbnail_dir = Path("static/thumbnails")
        self.metadata_file = "image_metadata.json"
        
        # Create directories
        self.web_dir.mkdir(exist_ok=True)
        self.thumbnail_dir.mkdir(exist_ok=True)
        
        # Image metadata
        self.image_metadata = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load existing metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.image_metadata = json.load(f)
                logger.info(f"Loaded {len(self.image_metadata)} images from metadata")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.image_metadata = {}
    
    def save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.image_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata for {len(self.image_metadata)} images")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def optimize_image(self, source_path, target_path, max_size=(800, 600)):
        """Optimize image for web display"""
        try:
            with Image.open(source_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save optimized image
                img.save(target_path, 'JPEG', quality=85, optimize=True)
                
                return True
        except Exception as e:
            logger.error(f"Error optimizing {source_path}: {e}")
            return False
    
    def create_thumbnail(self, source_path, thumbnail_path, size=(200, 150)):
        """Create thumbnail for grid display"""
        try:
            with Image.open(source_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Create a white background
                background = Image.new('RGB', size, (255, 255, 255))
                
                # Calculate position to center the image
                offset = ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
                background.paste(img, offset)
                
                # Save thumbnail
                background.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                
                return True
        except Exception as e:
            logger.error(f"Error creating thumbnail {source_path}: {e}")
            return False
    
    def scan_and_upload(self, max_images=50):
        """Scan source directory and upload images"""
        if not self.source_dir.exists():
            logger.error(f"Source directory {self.source_dir} does not exist")
            return
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.source_dir.rglob(f'*{ext}'))
            image_files.extend(self.source_dir.rglob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process images (limit to max_images for testing)
        processed_count = 0
        for image_path in image_files[:max_images]:
            try:
                # Create relative path for web
                relative_path = image_path.relative_to(self.source_dir)
                web_path = self.web_dir / relative_path
                thumbnail_path = self.thumbnail_dir / f"{relative_path.stem}_{relative_path.parent.name}.jpg"
                
                # Create web directory structure
                web_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if already processed
                image_id = f"{relative_path.parent.name}/{relative_path.name}"
                if image_id in self.image_metadata:
                    logger.debug(f"Already processed: {image_id}")
                    continue
                
                # Optimize and upload image
                if self.optimize_image(image_path, web_path):
                    # Create thumbnail
                    if self.create_thumbnail(image_path, thumbnail_path):
                        # Add to metadata
                        self.image_metadata[image_id] = {
                            'id': len(self.image_metadata),
                            'original_path': str(image_path),
                            'web_path': str(web_path),
                            'thumbnail_path': str(thumbnail_path),
                            'filename': str(relative_path),
                            'size': os.path.getsize(web_path),
                            'uploaded_at': time.time()
                        }
                        
                        processed_count += 1
                        logger.info(f"Uploaded: {image_id}")
                        
                        # Save metadata periodically
                        if processed_count % 10 == 0:
                            self.save_metadata()
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        # Save final metadata
        self.save_metadata()
        logger.info(f"Uploaded {processed_count} new images")
    
    def create_sample_images(self, count=20):
        """Create sample images for testing if no real images available"""
        logger.info(f"Creating {count} sample images for testing")
        
        for i in range(count):
            # Create sample image
            img = Image.new('RGB', (400, 300), color=(100 + i * 10, 150 + i * 5, 200 + i * 8))
            
            # Add text
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Create folder structure
            folder_name = f"L{(i // 5) + 1:02d}_V{(i % 5) + 1:03d}"
            image_name = f"{i:04d}.jpg"
            
            # Add text to image
            text = f"{folder_name}/{image_name}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (400 - text_width) // 2
            y = (300 - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
            
            # Save to web directory
            web_folder = self.web_dir / folder_name
            web_folder.mkdir(parents=True, exist_ok=True)
            
            web_path = web_folder / image_name
            img.save(web_path, 'JPEG', quality=85)
            
            # Create thumbnail
            thumbnail_path = self.thumbnail_dir / f"{folder_name}_{image_name}"
            self.create_thumbnail(web_path, thumbnail_path)
            
            # Add to metadata
            image_id = f"{folder_name}/{image_name}"
            self.image_metadata[image_id] = {
                'id': i,
                'original_path': str(web_path),
                'web_path': str(web_path),
                'thumbnail_path': str(thumbnail_path),
                'filename': image_id,
                'size': os.path.getsize(web_path),
                'uploaded_at': time.time(),
                'is_sample': True
            }
        
        self.save_metadata()
        logger.info(f"Created {count} sample images")

def main():
    """Main function"""
    uploader = ImageUploader()
    
    # Try to upload real images first
    logger.info("Scanning for real images...")
    uploader.scan_and_upload(max_images=100)
    
    # If no images found, create samples
    if len(uploader.image_metadata) == 0:
        logger.info("No real images found, creating sample images...")
        uploader.create_sample_images(20)
    
    logger.info(f"Total images available: {len(uploader.image_metadata)}")
    logger.info("Image upload completed!")

if __name__ == "__main__":
    main()




