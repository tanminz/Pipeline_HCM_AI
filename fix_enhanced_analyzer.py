#!/usr/bin/env python3
"""
Fix Enhanced Image Analyzer - Remove easyocr dependency
Sửa lỗi easyocr trong enhanced_image_analyzer.py
"""

def fix_enhanced_analyzer():
    """Sửa lỗi easyocr trong enhanced_image_analyzer.py"""
    
    # Đọc file enhanced_image_analyzer.py
    with open('enhanced_image_analyzer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Thay thế easyocr import bằng comment
    content = content.replace(
        'import easyocr',
        '# import easyocr  # Disabled for compatibility'
    )
    
    # Thay thế easyocr initialization
    content = content.replace(
        'self.ocr_reader = easyocr.Reader([\'en\', \'vi\'])',
        'self.ocr_reader = None  # easyocr.Reader([\'en\', \'vi\'])  # Disabled'
    )
    
    # Sửa OCR method để tránh lỗi
    ocr_method = '''
    def _extract_text_ocr(self, image):
        """Extract text from image using OCR"""
        try:
            if self.ocr_reader is None:
                return {"text": "", "confidence": 0.0}
            
            results = self.ocr_reader.readtext(np.array(image))
            if results:
                text = " ".join([result[1] for result in results])
                confidence = np.mean([result[2] for result in results])
                return {"text": text, "confidence": confidence}
            return {"text": "", "confidence": 0.0}
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"text": "", "confidence": 0.0}
'''
    
    # Tìm và thay thế OCR method
    import re
    pattern = r'def _extract_text_ocr\(self, image\):.*?return \{"text": "", "confidence": 0\.0\}'
    content = re.sub(pattern, ocr_method, content, flags=re.DOTALL)
    
    # Ghi lại file
    with open('enhanced_image_analyzer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Enhanced Image Analyzer fixed - easyocr dependency disabled")
    print("🔄 OCR functionality will return empty results but won't cause errors")

if __name__ == "__main__":
    fix_enhanced_analyzer()



