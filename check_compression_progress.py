#!/usr/bin/env python3
"""
Script kiểm tra tiến trình nén ảnh
"""

import os
from pathlib import Path
import time

def get_directory_size(directory):
    """Tính tổng dung lượng thư mục"""
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

def main():
    directories = [
        "static/images",
        "extracted_data"
    ]
    
    print("=== KIỂM TRA DUNG LƯỢNG SAU NÉN ===\n")
    
    for directory in directories:
        if os.path.exists(directory):
            size_bytes, file_count = get_directory_size(directory)
            size_gb = size_bytes / (1024**3)
            
            print(f"📁 {directory}:")
            print(f"   📊 Số file: {file_count:,}")
            print(f"   💾 Dung lượng: {size_gb:.2f} GB")
            print()
        else:
            print(f"❌ Thư mục không tồn tại: {directory}")

if __name__ == "__main__":
    main()
