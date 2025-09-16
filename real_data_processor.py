#!/usr/bin/env python3
"""
Real Data Processor - Process actual competition data from zip files
Handles keyframes L21-L30 and related metadata
"""

import os
import zipfile
import json
import logging
from pathlib import Path
from PIL import Image
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataProcessor:
    def __init__(self):
        self.data_dir = Path("D:/images")  # Where the zip files are located
        self.extract_dir = Path("extracted_data")
        self.web_dir = Path("static/images")
        self.thumbnail_dir = Path("static/thumbnails")
        self.metadata_file = "real_image_metadata.json"
        
        # Create directories
        self.extract_dir.mkdir(exist_ok=True)
        self.web_dir.mkdir(exist_ok=True)
        self.thumbnail_dir.mkdir(exist_ok=True)
        
        # Expected zip files
        self.keyframe_zips = [
            "Keyframes_L21.zip", "Keyframes_L22.zip", "Keyframes_L23.zip", 
            "Keyframes_L24.zip", "Keyframes_L25.zip", "Keyframes_L26_a.zip",
            "Keyframes_L26_b.zip", "Keyframes_L26_c.zip", "Keyframes_L26_d.zip",
            "Keyframes_L26_e.zip", "Keyframes_L27.zip", "Keyframes_L28.zip",
            "Keyframes_L29.zip", "Keyframes_L30.zip"
        ]
        
        self.video_zips = [
            "Videos_L21_a.zip", "Videos_L22_a.zip", "Videos_L23_a.zip",
            "Videos_L24_a.zip", "Videos_L25_a.zip", "Videos_L25_a1.zip",
            "Videos_L25_b.zip", "Videos_L26_a.zip", "Videos_L26_b.zip",
            "Videos_L26_c.zip", "Videos_L26_d.zip", "Videos_L26_e.zip",
            "Videos_L27_a.zip", "Videos_L28_a.zip", "Videos_L29_a.zip",
            "Videos_L30_a.zip"
        ]
        
        self.metadata_zips = [
            "clip-features-32-aic25-b1.zip",
            "map-keyframes-aic25-b1.zip", 
            "media-info-aic25-b1.zip",
            "objects-aic25-b1.zip"
        ]
    
    def check_data_availability(self):
        """Check which data files are available"""
        available_files = []
        missing_files = []
        
        all_files = self.keyframe_zips + self.video_zips + self.metadata_zips
        
        for file in all_files:
            file_path = self.data_dir / file
            if file_path.exists():
                available_files.append(file)
                logger.info(f"Found: {file}")
            else:
                missing_files.append(file)
                logger.warning(f"Missing: {file}")
        
        return available_files, missing_files
    
    def extract_keyframes(self):
        """Extract keyframe images from zip files"""
        logger.info("Extracting keyframes...")
        
        extracted_count = 0
        for zip_file in self.keyframe_zips:
            zip_path = self.data_dir / zip_file
            if not zip_path.exists():
                logger.warning(f"Keyframe zip not found: {zip_file}")
                continue
            
            try:
                extract_path = self.extract_dir / zip_file.replace('.zip', '')
                extract_path.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                logger.info(f"Extracted: {zip_file}")
                extracted_count += 1
                
            except Exception as e:
                logger.error(f"Error extracting {zip_file}: {e}")
        
        logger.info(f"Extracted {extracted_count} keyframe zip files")
        return extracted_count > 0
    
    def extract_metadata(self):
        """Extract metadata files"""
        logger.info("Extracting metadata...")
        
        for zip_file in self.metadata_zips:
            zip_path = self.data_dir / zip_file
            if not zip_path.exists():
                logger.warning(f"Metadata zip not found: {zip_file}")
                continue
            
            try:
                extract_path = self.extract_dir / "metadata"
                extract_path.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                logger.info(f"Extracted metadata: {zip_file}")
                
            except Exception as e:
                logger.error(f"Error extracting metadata {zip_file}: {e}")
    
    def scan_keyframes(self):
        """Scan extracted keyframes and create metadata"""
        logger.info("Scanning keyframes...")
        
        image_metadata = {}
        image_id = 0
        
        # Scan all extracted keyframe directories
        for item in self.extract_dir.iterdir():
            if item.is_dir() and item.name.startswith('Keyframes_'):
                logger.info(f"Scanning directory: {item.name}")
                
                # Find all image files in this directory
                for image_file in item.rglob('*.jpg'):
                    try:
                        # Create relative path for web
                        relative_path = image_file.relative_to(self.extract_dir)
                        web_path = self.web_dir / relative_path
                        thumbnail_path = self.thumbnail_dir / f"{relative_path.stem}_{relative_path.parent.name}.jpg"
                        
                        # Create web directory structure
                        web_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy image to web directory
                        shutil.copy2(image_file, web_path)
                        
                        # Create thumbnail
                        self.create_thumbnail(image_file, thumbnail_path)
                        
                        # Add to metadata
                        image_id_str = f"{relative_path.parent.name}/{relative_path.name}"
                        image_metadata[image_id_str] = {
                            'id': image_id,
                            'original_path': str(image_file),
                            'web_path': str(web_path),
                            'thumbnail_path': str(thumbnail_path),
                            'filename': str(relative_path),
                            'size': os.path.getsize(web_path),
                            'source_zip': item.name,
                            'is_real_data': True
                        }
                        
                        image_id += 1
                        
                        if image_id % 100 == 0:
                            logger.info(f"Processed {image_id} images...")
                    
                    except Exception as e:
                        logger.error(f"Error processing {image_file}: {e}")
        
        # Save metadata
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(image_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Total images processed: {len(image_metadata)}")
        return image_metadata
    
    def create_thumbnail(self, source_path, thumbnail_path, size=(200, 150)):
        """Create thumbnail for image"""
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
    
    def load_clip_features(self):
        """Load CLIP features if available"""
        clip_features_path = self.extract_dir / "metadata" / "clip-features-32-aic25-b1.npy"
        if clip_features_path.exists():
            try:
                import numpy as np
                features = np.load(clip_features_path)
                logger.info(f"Loaded CLIP features: {features.shape}")
                return features
            except Exception as e:
                logger.error(f"Error loading CLIP features: {e}")
        
        return None
    
    def load_keyframe_mapping(self):
        """Load keyframe mapping if available"""
        mapping_path = self.extract_dir / "metadata" / "map-keyframes-aic25-b1.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                logger.info(f"Loaded keyframe mapping: {len(mapping)} entries")
                return mapping
            except Exception as e:
                logger.error(f"Error loading keyframe mapping: {e}")
        
        return None
    
    def process_real_data(self):
        """Main function to process real competition data"""
        logger.info("Starting real data processing...")
        
        # Check data availability
        available_files, missing_files = self.check_data_availability()
        logger.info(f"Available files: {len(available_files)}")
        logger.info(f"Missing files: {len(missing_files)}")
        
        if not available_files:
            logger.error("No data files found!")
            return False
        
        # Extract keyframes
        if not self.extract_keyframes():
            logger.error("Failed to extract keyframes!")
            return False
        
        # Extract metadata
        self.extract_metadata()
        
        # Scan and process keyframes
        image_metadata = self.scan_keyframes()
        
        # Load additional data
        clip_features = self.load_clip_features()
        keyframe_mapping = self.load_keyframe_mapping()
        
        logger.info("Real data processing completed!")
        logger.info(f"Total images: {len(image_metadata)}")
        
        return True

def main():
    """Main function"""
    processor = RealDataProcessor()
    success = processor.process_real_data()
    
    if success:
        print("✅ Real data processing completed successfully!")
        print("You can now run the Flask app with real competition data.")
    else:
        print("❌ Real data processing failed!")
        print("Please check if the zip files are available in D:/images")

if __name__ == "__main__":
    main()




