#!/usr/bin/env python3
"""
Script sửa lỗi đường dẫn ảnh trong web
"""

import os
import json
from pathlib import Path

def find_available_videos():
    """Tìm các video có sẵn trong extracted_data"""
    available_videos = []
    
    for keyframe_dir in Path("extracted_data").glob("Keyframes_*"):
        if keyframe_dir.is_dir() and (keyframe_dir / "keyframes").exists():
            for video_dir in (keyframe_dir / "keyframes").iterdir():
                if video_dir.is_dir():
                    video_name = video_dir.name
                    available_videos.append({
                        "video": video_name,
                        "path": str(video_dir),
                        "keyframe_dir": keyframe_dir.name
                    })
    
    return available_videos

def update_image_metadata():
    """Cập nhật metadata với đường dẫn ảnh đúng"""
    available_videos = find_available_videos()
    
    # Tạo mapping video -> đường dẫn
    video_mapping = {}
    for video_info in available_videos:
        video_mapping[video_info["video"]] = {
            "keyframe_dir": video_info["keyframe_dir"],
            "path": video_info["path"]
        }
    
    # Lưu mapping
    with open("video_mapping.json", "w", encoding="utf-8") as f:
        json.dump(video_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã tạo mapping cho {len(video_mapping)} videos")
    return video_mapping

def test_image_access():
    """Test truy cập ảnh"""
    available_videos = find_available_videos()
    
    print("🔍 Test truy cập ảnh:")
    for i, video_info in enumerate(available_videos[:5]):  # Test 5 video đầu
        video_path = Path(video_info["path"])
        images = list(video_path.glob("*.jpg"))
        
        if images:
            test_image = images[0]
            print(f"{i+1}. {video_info['video']}: {len(images)} ảnh")
            print(f"   Test: {test_image.name} - {'✅ OK' if test_image.exists() else '❌ Lỗi'}")
        else:
            print(f"{i+1}. {video_info['video']}: ❌ Không có ảnh")

def main():
    print("🔧 SỬA LỖI ĐƯỜNG DẪN ẢNH")
    print("=" * 40)
    
    # Tìm videos có sẵn
    available_videos = find_available_videos()
    print(f"📊 Tìm thấy {len(available_videos)} videos")
    
    # Test truy cập ảnh
    test_image_access()
    
    # Cập nhật metadata
    video_mapping = update_image_metadata()
    
    print(f"\n✅ HOÀN THÀNH!")
    print(f"📁 Có {len(video_mapping)} videos sẵn sàng")
    print("🌐 Web sẽ tìm ảnh từ các video có sẵn")

if __name__ == "__main__":
    main()
