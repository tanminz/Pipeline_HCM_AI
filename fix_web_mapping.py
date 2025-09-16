#!/usr/bin/env python3
"""
Script sửa mapping trong web để tìm đúng ảnh
"""

import json
import os
from pathlib import Path

def create_video_alias_mapping():
    """Tạo mapping alias cho video"""
    print("🔧 Tạo video alias mapping...")
    
    # Mapping từ tên cũ sang tên mới
    alias_mapping = {
        "L03_V004": "K03_V004",  # Web tìm L03_V004 nhưng thực tế có K03_V004
        "L03_V001": "K03_V001",
        "L03_V002": "K03_V002", 
        "L03_V003": "K03_V003",
        "L03_V005": "K03_V005",
        # Thêm các mapping khác nếu cần
    }
    
    # Lưu mapping
    with open("video_alias_mapping.json", "w", encoding="utf-8") as f:
        json.dump(alias_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã tạo {len(alias_mapping)} alias mappings")
    return alias_mapping

def update_web_config():
    """Cập nhật cấu hình web"""
    print("🌐 Cập nhật cấu hình web...")
    
    # Tạo file config cho web
    web_config = {
        "image_base_path": "extracted_data",
        "video_alias_mapping": "video_alias_mapping.json",
        "supported_video_formats": ["K03_", "K01_", "K02_", "L21_", "L22_", "L23_", "L24_", "L25_", "L26_", "L27_", "L28_", "L29_", "L30_"],
        "fallback_search": True
    }
    
    with open("web_config.json", "w", encoding="utf-8") as f:
        json.dump(web_config, f, indent=2, ensure_ascii=False)
    
    print("✅ Đã cập nhật web config")

def test_image_access():
    """Test truy cập ảnh"""
    print("🔍 Test truy cập ảnh...")
    
    # Test ảnh K03_V004/0013.jpg
    test_path = Path("extracted_data/Keyframes_L03/keyframes/K03_V004/0013.jpg")
    
    if test_path.exists():
        print(f"✅ Ảnh có sẵn: {test_path}")
        print(f"   Kích thước: {test_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"❌ Ảnh không tồn tại: {test_path}")
        
        # Tìm ảnh thay thế
        alt_path = Path("extracted_data/Keyframes_L03/keyframes/K03_V004/001.jpg")
        if alt_path.exists():
            print(f"✅ Ảnh thay thế: {alt_path}")

def main():
    print("🔧 SỬA WEB MAPPING ĐỂ TÌM ĐÚNG ẢNH")
    print("=" * 50)
    
    # Tạo alias mapping
    alias_mapping = create_video_alias_mapping()
    
    # Cập nhật web config
    update_web_config()
    
    # Test truy cập ảnh
    test_image_access()
    
    print("\n✅ HOÀN THÀNH!")
    print("🌐 Web sẽ sử dụng alias mapping để tìm ảnh")
    print("📝 Cần cập nhật code web để sử dụng mapping này")

if __name__ == "__main__":
    main()
