#!/usr/bin/env python3
"""
Script tích hợp data mới nhất từ D:/images
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
        logging.FileHandler('latest_integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class LatestDataIntegrator:
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
    
    def copy_keyframes_from_d_images(self):
        """Copy keyframes từ D:/images/keyframes"""
        logging.info("🔄 Copy keyframes từ D:/images/keyframes...")
        
        source_keyframes = self.source_dir / "keyframes"
        if not source_keyframes.exists():
            logging.warning("Không tìm thấy D:/images/keyframes")
            return
        
        # Tạo thư mục Keyframes_L03 nếu chưa có
        target_l03 = self.target_dir / "Keyframes_L03"
        target_l03.mkdir(exist_ok=True)
        target_keyframes = target_l03 / "keyframes"
        target_keyframes.mkdir(exist_ok=True)
        
        # Copy các video K03_*
        k03_videos = [d for d in source_keyframes.iterdir() if d.is_dir() and d.name.startswith("K03_")]
        
        logging.info(f"Tìm thấy {len(k03_videos)} video K03_*")
        
        for video_dir in tqdm(k03_videos, desc="Copy K03 videos"):
            target_video = target_keyframes / video_dir.name
            
            if target_video.exists():
                logging.info(f"Video đã tồn tại: {video_dir.name}")
                continue
            
            try:
                shutil.copytree(video_dir, target_video)
                size, count = self.get_directory_size(target_video)
                logging.info(f"✅ {video_dir.name}: {count} files, {size/(1024**2):.2f} MB")
            except Exception as e:
                logging.error(f"❌ Lỗi copy {video_dir.name}: {e}")
    
    def copy_additional_keyframes(self):
        """Copy các keyframes khác còn thiếu"""
        logging.info("🔄 Copy keyframes bổ sung...")
        
        source_keyframes = self.source_dir / "keyframes"
        if not source_keyframes.exists():
            return
        
        # Tìm các thư mục keyframes còn thiếu
        existing_dirs = {d.name for d in self.target_dir.iterdir() if d.is_dir() and d.name.startswith("Keyframes_")}
        
        for keyframe_dir in source_keyframes.iterdir():
            if not keyframe_dir.is_dir():
                continue
            
            # Tạo tên thư mục target
            if keyframe_dir.name.startswith("K"):
                target_name = f"Keyframes_{keyframe_dir.name[:3]}"  # K03 -> Keyframes_K03
            elif keyframe_dir.name.startswith("L"):
                target_name = f"Keyframes_{keyframe_dir.name[:3]}"  # L21 -> Keyframes_L21
            else:
                continue
            
            if target_name in existing_dirs:
                continue
            
            # Tạo thư mục mới
            target_dir = self.target_dir / target_name
            target_keyframes = target_dir / "keyframes"
            target_keyframes.mkdir(parents=True, exist_ok=True)
            
            # Copy video đầu tiên để test
            videos = [d for d in keyframe_dir.iterdir() if d.is_dir()]
            if videos:
                test_video = videos[0]
                target_video = target_keyframes / test_video.name
                
                try:
                    shutil.copytree(test_video, target_video)
                    size, count = self.get_directory_size(target_video)
                    logging.info(f"✅ Tạo {target_name}: {count} files, {size/(1024**2):.2f} MB")
                except Exception as e:
                    logging.error(f"❌ Lỗi tạo {target_name}: {e}")
    
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
    
    def integrate_latest(self):
        """Tích hợp data mới nhất"""
        logging.info("🚀 TÍCH HỢP DATA MỚI NHẤT")
        logging.info("=" * 50)
        
        # Copy keyframes K03
        self.copy_keyframes_from_d_images()
        
        # Copy keyframes bổ sung
        self.copy_additional_keyframes()
        
        # Cập nhật mapping
        video_mapping = self.update_video_mapping()
        
        # Thống kê cuối
        total_size, total_files = self.get_directory_size(self.target_dir)
        logging.info("=" * 50)
        logging.info(f"🎉 HOÀN THÀNH TÍCH HỢP DATA MỚI!")
        logging.info(f"📊 Tổng files: {total_files:,}")
        logging.info(f"💾 Tổng dung lượng: {total_size/(1024**3):.2f} GB")
        logging.info(f"🎬 Tổng videos: {len(video_mapping)}")

def main():
    integrator = LatestDataIntegrator()
    integrator.integrate_latest()

if __name__ == "__main__":
    main()
