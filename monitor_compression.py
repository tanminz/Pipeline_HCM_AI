#!/usr/bin/env python3
"""
Script theo dõi tiến trình nén ảnh real-time
"""

import os
import time
from pathlib import Path

def get_directory_stats(directory):
    """Lấy thống kê thư mục"""
    if not os.path.exists(directory):
        return None, None, None
    
    total_size = 0
    file_count = 0
    jpg_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
                
                if file.lower().endswith(('.jpg', '.jpeg')):
                    jpg_count += 1
            except:
                pass
    
    return total_size, file_count, jpg_count

def format_size(size_bytes):
    """Format kích thước"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"

def main():
    print("🔍 THEO DÕI TIẾN TRÌNH NÉN ẢNH")
    print("=" * 50)
    
    # Dung lượng ban đầu
    original_sizes = {
        "static/images": 41.78 * 1024**3,  # 41.78 GB
        "extracted_data": 21.07 * 1024**3   # 21.07 GB
    }
    
    while True:
        print(f"\n⏰ {time.strftime('%H:%M:%S')}")
        print("-" * 30)
        
        total_current = 0
        total_original = sum(original_sizes.values())
        
        for directory in ["static/images", "extracted_data"]:
            size, file_count, jpg_count = get_directory_stats(directory)
            
            if size is not None:
                original = original_sizes.get(directory, size)
                compression_ratio = (1 - size/original) * 100 if original > 0 else 0
                
                print(f"📁 {directory}:")
                print(f"   📊 Files: {file_count:,} (JPG: {jpg_count:,})")
                print(f"   💾 Size: {format_size(size)}")
                print(f"   📉 Compression: {compression_ratio:.1f}%")
                
                total_current += size
            else:
                print(f"❌ {directory}: Not found")
        
        # Tổng kết
        total_compression = (1 - total_current/total_original) * 100 if total_original > 0 else 0
        space_saved = total_original - total_current
        
        print(f"\n🎯 TỔNG KẾT:")
        print(f"   💾 Current: {format_size(total_current)}")
        print(f"   📉 Total compression: {total_compression:.1f}%")
        print(f"   💰 Space saved: {format_size(space_saved)}")
        
        time.sleep(30)  # Cập nhật mỗi 30 giây

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Dừng theo dõi.")
