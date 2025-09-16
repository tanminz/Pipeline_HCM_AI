#!/usr/bin/env python3
"""
Script nén ảnh đơn giản cho Windows
"""

import os
import sys
from PIL import Image
from pathlib import Path
import time

def compress_image(input_path, quality=80, max_size=(1920, 1080)):
    """Nén một ảnh"""
    try:
        with Image.open(input_path) as img:
            # Chuyển sang RGB nếu cần
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize nếu quá lớn
            if img.width > max_size[0] or img.height > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Lưu với chất lượng thấp hơn
            img.save(input_path, 'JPEG', quality=quality, optimize=True)
            return True
    except Exception as e:
        print(f"Error compressing {input_path}: {e}")
        return False

def compress_directory(directory, quality=80, max_size=(1920, 1080)):
    """Nén tất cả ảnh trong thư mục"""
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
        
        if compress_image(img_file, quality, max_size):
            compressed_count += 1
    
    # Tính dung lượng sau
    total_size_after = sum(f.stat().st_size for f in image_files)
    compression_ratio = (1 - total_size_after/total_size_before) * 100
    
    print(f"\nCompression completed!")
    print(f"Compressed: {compressed_count}/{len(image_files)} files")
    print(f"Size before: {total_size_before / (1024*1024*1024):.2f} GB")
    print(f"Size after: {total_size_after / (1024*1024*1024):.2f} GB")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    print(f"Space saved: {(total_size_before - total_size_after) / (1024*1024*1024):.2f} GB")
    print(f"Time taken: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_images_simple.py <directory> [quality] [max_width] [max_height]")
        print("Example: python compress_images_simple.py static/images 80 1920 1080")
        sys.exit(1)
    
    directory = sys.argv[1]
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    max_width = int(sys.argv[3]) if len(sys.argv) > 3 else 1920
    max_height = int(sys.argv[4]) if len(sys.argv) > 4 else 1080
    
    print(f"Compressing images in: {directory}")
    print(f"Quality: {quality}")
    print(f"Max size: {max_width}x{max_height}")
    print()
    
    compress_directory(directory, quality, (max_width, max_height))
