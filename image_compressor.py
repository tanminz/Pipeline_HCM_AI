#!/usr/bin/env python3
"""
Script nén ảnh tự động để giảm dung lượng
Hỗ trợ: JPG, PNG, WebP
"""

import os
import sys
from PIL import Image, ImageOps
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compression.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ImageCompressor:
    def __init__(self, quality=85, max_width=1920, max_height=1080, 
                 backup_originals=True, output_format='JPEG'):
        self.quality = quality
        self.max_width = max_width
        self.max_height = max_height
        self.backup_originals = backup_originals
        self.output_format = output_format
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def get_file_size_mb(self, file_path):
        """Lấy kích thước file tính bằng MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def compress_image(self, input_path, output_path=None):
        """Nén một ảnh"""
        try:
            with Image.open(input_path) as img:
                # Chuyển sang RGB nếu cần (cho JPEG)
                if self.output_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                    # Tạo background trắng cho ảnh có alpha
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize nếu ảnh quá lớn
                if img.width > self.max_width or img.height > self.max_height:
                    img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
                
                # Tối ưu hóa ảnh
                img = ImageOps.exif_transpose(img)  # Xử lý EXIF orientation
                
                # Lưu ảnh đã nén
                if output_path is None:
                    output_path = input_path
                
                save_kwargs = {
                    'format': self.output_format,
                    'optimize': True,
                    'quality': self.quality
                }
                
                if self.output_format == 'JPEG':
                    save_kwargs['progressive'] = True
                
                img.save(output_path, **save_kwargs)
                
                return True
                
        except Exception as e:
            logging.error(f"Lỗi khi nén {input_path}: {str(e)}")
            return False
    
    def compress_directory(self, input_dir, output_dir=None, 
                          create_backup=True, dry_run=False):
        """Nén toàn bộ thư mục"""
        input_path = Path(input_dir)
        if not input_path.exists():
            logging.error(f"Thư mục không tồn tại: {input_dir}")
            return
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Tìm tất cả file ảnh
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            logging.warning(f"Không tìm thấy file ảnh nào trong {input_dir}")
            return
        
        logging.info(f"Tìm thấy {len(image_files)} file ảnh")
        
        # Tính tổng dung lượng ban đầu
        total_original_size = sum(self.get_file_size_mb(f) for f in image_files)
        logging.info(f"Tổng dung lượng ban đầu: {total_original_size:.2f} MB")
        
        if dry_run:
            logging.info("Chế độ dry-run: Không thực hiện nén")
            return
        
        # Tạo backup nếu cần
        if create_backup and self.backup_originals:
            backup_dir = input_path.parent / f"{input_path.name}_backup"
            if not backup_dir.exists():
                logging.info(f"Tạo backup tại: {backup_dir}")
                shutil.copytree(input_path, backup_dir)
        
        # Nén từng file
        compressed_count = 0
        total_compressed_size = 0
        
        for img_file in tqdm(image_files, desc="Đang nén ảnh"):
            try:
                original_size = self.get_file_size_mb(img_file)
                
                # Tạo đường dẫn output
                if output_dir == input_dir:
                    output_file = img_file
                else:
                    relative_path = img_file.relative_to(input_path)
                    output_file = Path(output_dir) / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Nén ảnh
                if self.compress_image(img_file, output_file):
                    compressed_size = self.get_file_size_mb(output_file)
                    total_compressed_size += compressed_size
                    compressed_count += 1
                    
                    compression_ratio = (1 - compressed_size/original_size) * 100
                    logging.debug(f"{img_file.name}: {original_size:.2f}MB -> {compressed_size:.2f}MB ({compression_ratio:.1f}% giảm)")
                
            except Exception as e:
                logging.error(f"Lỗi khi xử lý {img_file}: {str(e)}")
        
        # Báo cáo kết quả
        logging.info(f"Hoàn thành nén {compressed_count}/{len(image_files)} file")
        logging.info(f"Dung lượng sau nén: {total_compressed_size:.2f} MB")
        if total_original_size > 0:
            total_compression = (1 - total_compressed_size/total_original_size) * 100
            logging.info(f"Tổng tỷ lệ nén: {total_compression:.1f}%")
            logging.info(f"Tiết kiệm: {total_original_size - total_compressed_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Nén ảnh tự động')
    parser.add_argument('input_dir', help='Thư mục chứa ảnh cần nén')
    parser.add_argument('-o', '--output', help='Thư mục output (mặc định: ghi đè file gốc)')
    parser.add_argument('-q', '--quality', type=int, default=85, 
                       help='Chất lượng nén (1-100, mặc định: 85)')
    parser.add_argument('-w', '--max-width', type=int, default=1920,
                       help='Chiều rộng tối đa (mặc định: 1920)')
    parser.add_argument('--max-height', type=int, default=1080,
                       help='Chiều cao tối đa (mặc định: 1080)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Không tạo backup file gốc')
    parser.add_argument('--dry-run', action='store_true',
                       help='Chỉ hiển thị thông tin, không nén thực tế')
    parser.add_argument('--format', choices=['JPEG', 'PNG', 'WebP'], 
                       default='JPEG', help='Định dạng output (mặc định: JPEG)')
    
    args = parser.parse_args()
    
    # Tạo compressor
    compressor = ImageCompressor(
        quality=args.quality,
        max_width=args.max_width,
        max_height=args.max_height,
        backup_originals=not args.no_backup,
        output_format=args.format
    )
    
    # Thực hiện nén
    compressor.compress_directory(
        args.input_dir,
        args.output,
        create_backup=not args.no_backup,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
