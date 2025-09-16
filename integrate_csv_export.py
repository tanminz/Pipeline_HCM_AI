#!/usr/bin/env python3
"""
Integrate CSV Export System into app.py
Tích hợp hệ thống xuất CSV vào app.py
"""

import re

def integrate_csv_export():
    """Tích hợp CSV export system vào app.py"""
    
    # Đọc file app.py hiện tại
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Thêm import cho CSV export system
    import_statement = '''
from csv_export_system import create_csv_export_routes
'''
    
    # Tìm vị trí để thêm import
    if 'from csv_export_system import create_csv_export_routes' not in content:
        # Thêm import sau các import khác
        content = content.replace(
            'from flask import Flask, render_template, request, jsonify, send_file',
            'from flask import Flask, render_template, request, jsonify, send_file\nfrom csv_export_system import create_csv_export_routes'
        )
    
    # Thêm khởi tạo CSV export system sau khi khởi tạo enhanced_image_analyzer
    csv_init_code = '''
    # Initialize CSV Export System
    csv_export_system = None
    if enhanced_image_analyzer is not None:
        try:
            csv_export_system = create_csv_export_routes(app, enhanced_image_analyzer)
            logger.info("✅ CSV Export System initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CSV Export System: {e}")
            csv_export_system = None
'''
    
    # Tìm vị trí để thêm CSV export system initialization
    if 'csv_export_system = None' not in content:
        # Thêm sau enhanced_image_analyzer initialization
        content = content.replace(
            'logger.info(f"🎯 System initialized with {total_images} images")',
            'logger.info(f"🎯 System initialized with {total_images} images")' + csv_init_code
        )
    
    # Thêm global variable cho csv_export_system
    if 'csv_export_system = None' not in content:
        # Thêm vào phần global variables
        content = content.replace(
            'enhanced_image_analyzer = None',
            'enhanced_image_analyzer = None\ncsv_export_system = None'
        )
    
    # Ghi lại file app.py
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ CSV Export System integrated successfully!")
    print("🔄 Please restart the server to apply changes")

if __name__ == "__main__":
    integrate_csv_export()



