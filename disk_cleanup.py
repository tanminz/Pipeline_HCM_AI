#!/usr/bin/env python3
"""
Script dọn dẹp ổ đĩa để giải phóng dung lượng
"""

import os
import shutil
import tempfile
from pathlib import Path

def cleanup_windows_temp():
    """Dọn dẹp thư mục temp của Windows"""
    print("🧹 Dọn dẹp Windows temp...")
    
    temp_dirs = [
        os.environ.get('TEMP', ''),
        os.environ.get('TMP', ''),
        r'C:\Windows\Temp',
        r'C:\Windows\Prefetch'
    ]
    
    total_freed = 0
    
    for temp_dir in temp_dirs:
        if temp_dir and os.path.exists(temp_dir):
            try:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            size = os.path.getsize(file_path)
                            os.remove(file_path)
                            total_freed += size
                        except:
                            pass
                print(f"✅ Dọn dẹp: {temp_dir}")
            except:
                pass
    
    print(f"💾 Giải phóng: {total_freed / (1024*1024):.1f} MB")

def cleanup_python_cache():
    """Dọn dẹp Python cache"""
    print("🐍 Dọn dẹp Python cache...")
    
    freed = 0
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    for file in os.listdir(cache_path):
                        file_path = os.path.join(cache_path, file)
                        freed += os.path.getsize(file_path)
                    shutil.rmtree(cache_path)
                except:
                    pass
    
    print(f"💾 Giải phóng: {freed / (1024*1024):.1f} MB")

def cleanup_log_files():
    """Dọn dẹp file log"""
    print("📝 Dọn dẹp file log...")
    
    log_patterns = ['*.log', '*.tmp', '*.temp', '*.bak']
    freed = 0
    
    for pattern in log_patterns:
        for file_path in Path('.').rglob(pattern):
            try:
                freed += file_path.stat().st_size
                file_path.unlink()
            except:
                pass
    
    print(f"💾 Giải phóng: {freed / (1024*1024):.1f} MB")

def cleanup_large_files():
    """Tìm và xóa file lớn không cần thiết"""
    print("🔍 Tìm file lớn không cần thiết...")
    
    large_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > 100 * 1024 * 1024:  # > 100MB
                    large_files.append((file_path, size))
            except:
                pass
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print("📊 Top 10 file lớn nhất:")
    for i, (file_path, size) in enumerate(large_files[:10]):
        print(f"{i+1}. {file_path}: {size/(1024*1024):.1f} MB")

def main():
    print("🧹 DỌN DẸP Ổ ĐĨA ĐỂ GIẢI PHÓNG DUNG LƯỢNG")
    print("=" * 50)
    
    cleanup_windows_temp()
    cleanup_python_cache()
    cleanup_log_files()
    cleanup_large_files()
    
    print("\n✅ HOÀN THÀNH DỌN DẸP!")

if __name__ == "__main__":
    main()
