#!/usr/bin/env python3
"""
Script tích hợp data mới vào hệ thống
"""

import os
import shutil
import json
from pathlib import Path
import logging
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DataIntegrator:
    def __init__(self, source_dir="d:/images", target_dir="extracted_data"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)
        
    def get_directory_size(self, directory):
        """Tính dung lượng thư mục"""
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except:
                    pass
        
        return total_size, file_count
    
    def copy_keyframes(self, keyframe_dirs):
        """Copy các thư mục keyframes mới"""
        logging.info("🔄 Bắt đầu copy keyframes...")
        
        for keyframe_dir in keyframe_dirs:
            source_path = self.source_dir / keyframe_dir
            if not source_path.exists():
                logging.warning(f"Thư mục không tồn tại: {source_path}")
                continue
            
            # Tạo tên thư mục mới
            new_name = keyframe_dir.replace("Keyframes_", "Keyframes_")
            target_path = self.target_dir / new_name
            
            if target_path.exists():
                logging.info(f"Thư mục đã tồn tại, bỏ qua: {target_path}")
                continue
            
            logging.info(f"Copy {source_path} -> {target_path}")
            
            try:
                shutil.copytree(source_path, target_path)
                size, count = self.get_directory_size(target_path)
                logging.info(f"✅ Hoàn thành: {count} files, {size/(1024**3):.2f} GB")
            except Exception as e:
                logging.error(f"❌ Lỗi copy {source_path}: {e}")
    
    def copy_metadata(self, metadata_dirs):
        """Copy metadata và mapping files"""
        logging.info("🔄 Bắt đầu copy metadata...")
        
        # Tạo thư mục metadata
        metadata_target = self.target_dir / "metadata"
        metadata_target.mkdir(exist_ok=True)
        
        for metadata_dir in metadata_dirs:
            source_path = self.source_dir / metadata_dir
            if not source_path.exists():
                continue
            
            # Copy vào thư mục metadata với tên mới
            if "media-info" in metadata_dir:
                target_name = metadata_dir.replace("media-info", "media-info")
            elif "map-keyframes" in metadata_dir:
                target_name = metadata_dir.replace("map-keyframes", "map-keyframes")
            elif "clip-features" in metadata_dir:
                target_name = metadata_dir.replace("clip-features", "clip-features")
            else:
                target_name = metadata_dir
            
            target_path = metadata_target / target_name
            
            if target_path.exists():
                logging.info(f"Metadata đã tồn tại: {target_path}")
                continue
            
            try:
                shutil.copytree(source_path, target_path)
                size, count = self.get_directory_size(target_path)
                logging.info(f"✅ Metadata: {count} files, {size/(1024**2):.2f} MB")
            except Exception as e:
                logging.error(f"❌ Lỗi copy metadata {source_path}: {e}")
    
    def copy_objects(self, objects_dirs):
        """Copy object detection data"""
        logging.info("🔄 Bắt đầu copy objects data...")
        
        objects_target = self.target_dir / "objects"
        objects_target.mkdir(exist_ok=True)
        
        for objects_dir in objects_dirs:
            source_path = self.source_dir / objects_dir
            if not source_path.exists():
                continue
            
            target_name = objects_dir.replace("objects", "objects")
            target_path = objects_target / target_name
            
            if target_path.exists():
                logging.info(f"Objects data đã tồn tại: {target_path}")
                continue
            
            try:
                shutil.copytree(source_path, target_path)
                size, count = self.get_directory_size(target_path)
                logging.info(f"✅ Objects: {count} files, {size/(1024**2):.2f} MB")
            except Exception as e:
                logging.error(f"❌ Lỗi copy objects {source_path}: {e}")
    
    def update_image_metadata(self):
        """Cập nhật file metadata tổng"""
        logging.info("🔄 Cập nhật image metadata...")
        
        # Tìm tất cả ảnh trong extracted_data
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(self.target_dir.rglob(f"*{ext}"))
            image_files.extend(self.target_dir.rglob(f"*{ext.upper()}"))
        
        # Tạo metadata mới
        metadata = {
            "total_images": len(image_files),
            "directories": [],
            "last_updated": str(Path().cwd())
        }
        
        # Thống kê theo thư mục
        dir_stats = {}
        for img_file in image_files:
            dir_name = img_file.parent.name
            if dir_name not in dir_stats:
                dir_stats[dir_name] = {"count": 0, "size": 0}
            dir_stats[dir_name]["count"] += 1
            dir_stats[dir_name]["size"] += img_file.stat().st_size
        
        metadata["directories"] = dir_stats
        
        # Lưu metadata
        metadata_file = self.target_dir / "image_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Đã cập nhật metadata: {len(image_files)} ảnh")
    
    def integrate_all(self):
        """Tích hợp toàn bộ data mới"""
        logging.info("🚀 BẮT ĐẦU TÍCH HỢP DATA MỚI")
        logging.info("=" * 50)
        
        # Danh sách các thư mục cần copy
        keyframe_dirs = [f"Keyframes_K{i:02d}" for i in range(1, 15)]
        metadata_dirs = [
            "media-info-aic25-b1",
            "media-info-aic25-b2", 
            "map-keyframes-aic25-b1",
            "map-keyframes-b2",
            "clip-features-32-aic25-b1",
            "clip-features-32-aic25-b2"
        ]
        objects_dirs = ["objects-aic25-b2"]
        
        # Copy keyframes
        self.copy_keyframes(keyframe_dirs)
        
        # Copy metadata
        self.copy_metadata(metadata_dirs)
        
        # Copy objects
        self.copy_objects(objects_dirs)
        
        # Cập nhật metadata
        self.update_image_metadata()
        
        # Thống kê cuối
        total_size, total_files = self.get_directory_size(self.target_dir)
        logging.info("=" * 50)
        logging.info(f"🎉 HOÀN THÀNH TÍCH HỢP!")
        logging.info(f"📊 Tổng files: {total_files:,}")
        logging.info(f"💾 Tổng dung lượng: {total_size/(1024**3):.2f} GB")

def main():
    integrator = DataIntegrator()
    integrator.integrate_all()

if __name__ == "__main__":
    main()
