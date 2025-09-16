#!/usr/bin/env python3
"""
Script nén ảnh siêu mạnh để giảm dung lượng tối đa
"""

import os
import sys
from PIL import Image
from pathlib import Path
import time
import shutil

def aggressive_compress_image(input_path, quality=50, max_size=(1280, 720)):
    """Nén ảnh với cài đặt siêu mạnh"""
    try:
        with Image.open(input_path) as img:
            # Chuyển sang RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize mạnh hơn
            if img.width > max_size[0] or img.height > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Nén với chất lượng thấp
            img.save(input_path, 'JPEG', quality=quality, optimize=True, progressive=True)
            return True
    except Exception as e:
        print(f"Error compressing {input_path}: {e}")
        return False

def cleanup_temp_files():
    """Dọn dẹp file tạm và cache"""
    print("🧹 Dọn dẹp file tạm...")
    
    # Dọn dẹp Python cache
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"✅ Xóa cache: {cache_path}")
                except:
                    pass
    
    # Dọn dẹp file log cũ
    log_files = ['compression.log', 'data_integration.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                print(f"✅ Xóa log: {log_file}")
            except:
                pass

def compress_directory_aggressive(directory, quality=50, max_size=(1280, 720)):
    """Nén thư mục với cài đặt siêu mạnh"""
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    # Tìm tất cả file ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print("No image files found")
        return
    
    # Tính dung lượng ban đầu
    total_size_before = sum(f.stat().st_size for f in image_files)
    print(f"Total size before: {total_size_before / (1024*1024*1024):.2f} GB")
    
    # Nén từng file
    compressed_count = 0
    start_time = time.time()
    
    for i, img_file in enumerate(image_files):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(image_files)} files...")
        
        if aggressive_compress_image(img_file, quality, max_size):
            compressed_count += 1
    
    # Tính dung lượng sau
    total_size_after = sum(f.stat().st_size for f in image_files)
    compression_ratio = (1 - total_size_after/total_size_before) * 100
    
    print(f"\n🎯 NÉN SIÊU MẠNH HOÀN THÀNH!")
    print(f"Compressed: {compressed_count}/{len(image_files)} files")
    print(f"Size before: {total_size_before / (1024*1024*1024):.2f} GB")
    print(f"Size after: {total_size_after / (1024*1024*1024):.2f} GB")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    print(f"Space saved: {(total_size_before - total_size_after) / (1024*1024*1024):.2f} GB")
    print(f"Time taken: {time.time() - start_time:.1f} seconds")

def main():
    print("🚀 NÉN ẢNH SIÊU MẠNH ĐỂ GIẢM DUNG LƯỢNG TỐI ĐA")
    print("=" * 60)
    
    # Dọn dẹp file tạm trước
    cleanup_temp_files()
    
    # Nén extracted_data với cài đặt siêu mạnh
    print("\n📁 Nén extracted_data...")
    compress_directory_aggressive("extracted_data", quality=40, max_size=(1280, 720))
    
    print("\n🎉 HOÀN THÀNH NÉN SIÊU MẠNH!")
    print("Dung lượng đã được giảm tối đa để tích hợp đủ data mới.")

if __name__ == "__main__":
    main()
