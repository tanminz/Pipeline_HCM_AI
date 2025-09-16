#!/usr/bin/env python3
"""
Script tích hợp toàn bộ data từ D:/images
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
        logging.FileHandler('all_data_integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class AllDataIntegrator:
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
    
    def copy_all_keyframes(self):
        """Copy toàn bộ keyframes từ D:/images"""
        logging.info("🔄 Copy toàn bộ keyframes từ D:/images...")
        
        # Copy từ thư mục keyframes
        source_keyframes = self.source_dir / "keyframes"
        if source_keyframes.exists():
            self.copy_keyframes_directory(source_keyframes)
        
        # Copy từ các thư mục K01_V*, K02_V*, etc.
        for item in self.source_dir.iterdir():
            if item.is_dir() and item.name.startswith(("K01_", "K02_", "K03_", "L")):
                self.copy_individual_video(item)
    
    def copy_keyframes_directory(self, source_keyframes):
        """Copy từ thư mục keyframes chính"""
        logging.info(f"📁 Copy từ {source_keyframes}")
        
        for keyframe_dir in source_keyframes.iterdir():
            if not keyframe_dir.is_dir():
                continue
            
            # Tạo tên thư mục target
            if keyframe_dir.name.startswith("K"):
                target_name = f"Keyframes_{keyframe_dir.name[:3]}"
            elif keyframe_dir.name.startswith("L"):
                target_name = f"Keyframes_{keyframe_dir.name[:3]}"
            else:
                continue
            
            # Tạo thư mục target
            target_dir = self.target_dir / target_name
            target_keyframes = target_dir / "keyframes"
            target_keyframes.mkdir(parents=True, exist_ok=True)
            
            # Copy video
            target_video = target_keyframes / keyframe_dir.name
            
            if target_video.exists():
                logging.info(f"Video đã tồn tại: {keyframe_dir.name}")
                continue
            
            try:
                shutil.copytree(keyframe_dir, target_video)
                size, count = self.get_directory_size(target_video)
                logging.info(f"✅ {keyframe_dir.name}: {count} files, {size/(1024**2):.2f} MB")
            except Exception as e:
                logging.error(f"❌ Lỗi copy {keyframe_dir.name}: {e}")
    
    def copy_individual_video(self, video_dir):
        """Copy video riêng lẻ"""
        video_name = video_dir.name
        
        # Xác định thư mục target
        if video_name.startswith("K01_"):
            target_name = "Keyframes_K01"
        elif video_name.startswith("K02_"):
            target_name = "Keyframes_K02"
        elif video_name.startswith("K03_"):
            target_name = "Keyframes_K03"
        elif video_name.startswith("L"):
            target_name = f"Keyframes_{video_name[:3]}"
        else:
            return
        
        # Tạo thư mục target
        target_dir = self.target_dir / target_name
        target_keyframes = target_dir / "keyframes"
        target_keyframes.mkdir(parents=True, exist_ok=True)
        
        target_video = target_keyframes / video_name
        
        if target_video.exists():
            return
        
        try:
            shutil.copytree(video_dir, target_video)
            size, count = self.get_directory_size(target_video)
            logging.info(f"✅ {video_name}: {count} files, {size/(1024**2):.2f} MB")
        except Exception as e:
            logging.error(f"❌ Lỗi copy {video_name}: {e}")
    
    def copy_metadata(self):
        """Copy metadata"""
        logging.info("🔄 Copy metadata...")
        
        metadata_dirs = [
            "media-info-aic25-b1",
            "media-info-aic25-b2",
            "map-keyframes-aic25-b1", 
            "map-keyframes-b2",
            "clip-features-32-aic25-b1",
            "clip-features-32-aic25-b2",
            "objects-aic25-b2"
        ]
        
        metadata_target = self.target_dir / "metadata"
        metadata_target.mkdir(exist_ok=True)
        
        for metadata_dir in metadata_dirs:
            source_path = self.source_dir / metadata_dir
            if not source_path.exists():
                continue
            
            target_path = metadata_target / metadata_dir
            
            if target_path.exists():
                logging.info(f"Metadata đã tồn tại: {metadata_dir}")
                continue
            
            try:
                shutil.copytree(source_path, target_path)
                size, count = self.get_directory_size(target_path)
                logging.info(f"✅ {metadata_dir}: {count} files, {size/(1024**2):.2f} MB")
            except Exception as e:
                logging.error(f"❌ Lỗi copy metadata {metadata_dir}: {e}")
    
    def update_video_mapping(self):
        """Cập nhật mapping video"""
        logging.info("🔄 Cập nhật video mapping...")
        
        video_mapping = {}
        
        for keyframe_dir in self.target_dir.glob("Keyframes_*"):
            if keyframe_dir.is_dir() and (keyframe_dir / "keyframes").exists():
                for video_dir in (keyframe_dir / "keyframes").iterdir():
                    if video_dir.is_dir():
                        video_name = video_dir.name
                        video_mapping[video_name] = {
                            "keyframe_dir": keyframe_dir.name,
                            "path": str(video_dir.relative_to(self.target_dir))
                        }
        
        # Lưu mapping
        with open("video_mapping.json", "w", encoding="utf-8") as f:
            json.dump(video_mapping, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Đã cập nhật mapping cho {len(video_mapping)} videos")
        return video_mapping
    
    def integrate_all(self):
        """Tích hợp toàn bộ data"""
        logging.info("🚀 TÍCH HỢP TOÀN BỘ DATA")
        logging.info("=" * 50)
        
        # Copy keyframes
        self.copy_all_keyframes()
        
        # Copy metadata
        self.copy_metadata()
        
        # Cập nhật mapping
        video_mapping = self.update_video_mapping()
        
        # Thống kê cuối
        total_size, total_files = self.get_directory_size(self.target_dir)
        logging.info("=" * 50)
        logging.info(f"🎉 HOÀN THÀNH TÍCH HỢP TOÀN BỘ DATA!")
        logging.info(f"📊 Tổng files: {total_files:,}")
        logging.info(f"💾 Tổng dung lượng: {total_size/(1024**3):.2f} GB")
        logging.info(f"🎬 Tổng videos: {len(video_mapping)}")

def main():
    integrator = AllDataIntegrator()
    integrator.integrate_all()

if __name__ == "__main__":
    main()
