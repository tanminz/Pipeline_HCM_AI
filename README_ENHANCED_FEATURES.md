# Enhanced Image Analysis Features

## 🚀 Tính năng mới được thêm vào

Hệ thống tìm kiếm ảnh đã được nâng cấp với các tính năng phân tích ảnh nâng cao:

### 📝 OCR (Optical Character Recognition)
- **EasyOCR**: Nhận diện chữ tiếng Việt và tiếng Anh
- **TrOCR**: Nhận diện chữ viết tay
- **PaddleOCR**: OCR chính xác cao cho nhiều ngôn ngữ

### 🎯 Object Detection
- **DETR (Detection Transformer)**: Phát hiện đối tượng trong ảnh
- **ResNet-50**: Phân loại cảnh quan và đối tượng
- **Multi-label classification**: Phân loại nhiều đối tượng cùng lúc

### 🏞️ Scene Analysis
- **Scene Classification**: Phân loại loại cảnh (indoor/outdoor, office, street, etc.)
- **Color Analysis**: Phân tích màu sắc chủ đạo
- **Metadata Extraction**: Trích xuất thông tin ảnh

## 📦 Cài đặt

### 1. Cài đặt dependencies
```bash
python install_enhanced_dependencies.py
```

### 2. Hoặc cài đặt thủ công
```bash
pip install easyocr>=1.7.0
pip install PaddleOCR>=2.7.0
pip install pytesseract>=0.3.10
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
```

## 🧪 Kiểm tra tính năng

### Chạy test suite
```bash
python test_enhanced_analysis.py
```

### Test từng tính năng riêng lẻ
```python
# Test OCR
from enhanced_image_analyzer import analyze_single_image
result = analyze_single_image("path/to/image.jpg")
print(result['ocr_results']['combined_text'])

# Test object detection
print(result['objects']['main_objects'])

# Test scene classification
print(result['scene_classification']['scene_type'])
```

## 🔍 API Endpoints mới

### 1. Enhanced Analysis
```
GET /api/enhanced_analysis/<image_id>
```
Phân tích toàn diện ảnh bao gồm OCR, object detection, scene classification.

**Response:**
```json
{
  "image_id": 123,
  "analysis": {
    "ocr_results": {
      "combined_text": "Extracted text from image",
      "text_blocks": [...]
    },
    "objects": {
      "main_objects": ["person", "car", "building"],
      "object_count": 5
    },
    "scene_classification": {
      "scene_type": "outdoor",
      "confidence": 0.92
    },
    "text_summary": "Scene: outdoor | Objects: person, car | Text: Sample text",
    "search_keywords": ["outdoor", "person", "car", "text"]
  }
}
```

### 2. Search with Analysis
```
GET /api/search_with_analysis?q=cartoon boat&k=20
```
Tìm kiếm sử dụng phân tích ảnh nâng cao.

**Response:**
```json
{
  "results": [
    {
      "id": 123,
      "path": "/images/L21_V001/152.jpg",
      "filename": "L21_V001/152.jpg",
      "similarity": 0.85,
      "rank": 1,
      "analysis_summary": "Scene: outdoor | Objects: boat, water | Text: Cartoon boat",
      "detected_objects": ["boat", "water", "sky"],
      "extracted_text": "Cartoon boat on water"
    }
  ],
  "count": 1,
  "query": "cartoon boat",
  "strategy": "enhanced_analysis",
  "ai_engine": true,
  "engine_type": "enhanced_analyzer"
}
```

### 3. Analyzer Status
```
GET /api/analyzer_status
```
Kiểm tra trạng thái của enhanced analyzer.

**Response:**
```json
{
  "analyzer_available": true,
  "status": "ready",
  "statistics": {
    "total_analyzed": 1500,
    "successful_analyses": 1480,
    "text_detections": 320,
    "object_detections": 1200,
    "cache_size_mb": 45.2
  }
}
```

## 🎯 Cách sử dụng

### 1. Tìm kiếm bằng text trong ảnh
```
GET /api/search_with_analysis?q=HCMC&k=10
```
Tìm ảnh có chứa chữ "HCMC"

### 2. Tìm kiếm bằng đối tượng
```
GET /api/search_with_analysis?q=person car&k=20
```
Tìm ảnh có người và xe

### 3. Tìm kiếm bằng cảnh quan
```
GET /api/search_with_analysis?q=outdoor street&k=15
```
Tìm ảnh cảnh ngoài trời đường phố

### 4. Tìm kiếm kết hợp
```
GET /api/search_with_analysis?q=boat water cartoon&k=25
```
Tìm ảnh thuyền trên nước với phong cách hoạt hình

## 🔧 Cấu hình

### Cache Configuration
Phân tích ảnh được cache để tăng tốc độ:
- Cache được lưu trong thư mục `image_analysis_cache/`
- Tự động reload cache khi khởi động
- Cache theo hash của ảnh để tránh phân tích lại

### Model Configuration
```python
# Trong enhanced_image_analyzer.py
class EnhancedImageAnalyzer:
    def __init__(self, cache_dir="image_analysis_cache"):
        # Cấu hình cache
        self.cache_dir = cache_dir
        
        # Cấu hình models
        self.ocr_reader = easyocr.Reader(['en', 'vi'], gpu=torch.cuda.is_available())
        self.object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.image_classifier = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
```

## 📊 Performance

### Tốc độ phân tích
- **OCR**: ~2-5 giây/ảnh (tùy kích thước)
- **Object Detection**: ~3-8 giây/ảnh
- **Scene Classification**: ~1-3 giây/ảnh
- **Cache hit**: ~0.1 giây/ảnh

### Memory Usage
- **EasyOCR**: ~2GB RAM
- **DETR**: ~1.5GB RAM
- **ResNet-50**: ~1GB RAM
- **Cache**: ~50MB cho 1000 ảnh

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **ImportError: No module named 'easyocr'**
   ```bash
   pip install easyocr
   ```

2. **CUDA out of memory**
   ```python
   # Sử dụng CPU thay vì GPU
   reader = easyocr.Reader(['en', 'vi'], gpu=False)
   ```

3. **Model download failed**
   ```bash
   # Xóa cache và tải lại
   rm -rf ~/.cache/huggingface/
   ```

4. **Cache corruption**
   ```bash
   # Xóa cache
   rm -rf image_analysis_cache/
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Hoặc trong app.py
app.run(debug=True)
```

## 🚀 Tips & Tricks

### 1. Tối ưu tốc độ
- Sử dụng cache để tránh phân tích lại
- Batch processing cho nhiều ảnh
- Sử dụng GPU nếu có

### 2. Tăng độ chính xác
- Kết hợp nhiều model OCR
- Fine-tune threshold cho object detection
- Sử dụng ensemble methods

### 3. Query optimization
- Sử dụng từ khóa cụ thể
- Kết hợp text + object + scene
- Sử dụng synonyms và related terms

## 📈 Metrics & Monitoring

### Key Metrics
- **Text Detection Rate**: % ảnh có text được phát hiện
- **Object Detection Accuracy**: Độ chính xác phát hiện đối tượng
- **Search Relevance**: Độ liên quan của kết quả tìm kiếm
- **Response Time**: Thời gian phản hồi API

### Monitoring
```python
# Get statistics
stats = analyzer.get_analysis_statistics()
print(f"Cache hit rate: {stats['successful_analyses']/stats['total_analyzed']:.2%}")
```

## 🔮 Roadmap

### Planned Features
- [ ] Face recognition
- [ ] Action recognition
- [ ] Video analysis
- [ ] Multi-modal search
- [ ] Real-time analysis
- [ ] Custom model training

### Performance Improvements
- [ ] Model quantization
- [ ] Batch inference
- [ ] Distributed processing
- [ ] Edge deployment

---

**Lưu ý**: Các tính năng mới này yêu cầu nhiều tài nguyên hơn. Đảm bảo máy tính có đủ RAM (8GB+) và GPU (nếu có) để chạy hiệu quả.






