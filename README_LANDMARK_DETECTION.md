# 🏛️ Hệ thống Nhận diện Địa điểm Nổi tiếng

Hệ thống nhận diện địa điểm nổi tiếng cho cuộc thi HCMC AI Challenge, có thể nhận diện các địa điểm như Bitexco Financial Tower, Landmark 81, Chợ Bến Thành và nhiều địa điểm khác.

## 📋 Tính năng

- ✅ Nhận diện địa điểm nổi tiếng Việt Nam
- ✅ Hỗ trợ tiếng Việt và tiếng Anh
- ✅ Tích hợp với hệ thống phân tích hình ảnh hiện tại
- ✅ Huấn luyện model tùy chỉnh
- ✅ Cache kết quả phân tích
- ✅ Báo cáo đánh giá chi tiết

## 🏗️ Cấu trúc Dự án

```
Pipeline_HCM_AI/
├── landmark_detection_trainer.py      # Huấn luyện model
├── enhanced_landmark_detector.py      # Detector chính
├── integrate_landmark_detection.py    # Tích hợp vào hệ thống
├── train_landmarks.py                 # Script huấn luyện
├── test_landmark_detection.py         # Test hệ thống
├── landmark_data/                     # Dữ liệu huấn luyện
│   ├── bitexco/
│   ├── landmark81/
│   └── ben_thanh/
└── landmark_models/                   # Model đã huấn luyện
    ├── best_model.pth
    └── training_history.json
```

## 🚀 Cài đặt và Sử dụng

### 1. Cài đặt Dependencies

```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn
pip install opencv-python pillow
```

### 2. Tích hợp vào Hệ thống

```bash
python integrate_landmark_detection.py
```

Script này sẽ:
- Tích hợp landmark detection vào `enhanced_image_analyzer.py`
- Tạo cấu trúc thư mục cho dữ liệu huấn luyện
- Tạo script huấn luyện `train_landmarks.py`

### 3. Chuẩn bị Dữ liệu Huấn luyện

Tạo cấu trúc thư mục như sau:

```
landmark_data/
├── bitexco/
│   ├── bitexco_001.jpg
│   ├── bitexco_002.jpg
│   └── ...
├── landmark81/
│   ├── landmark81_001.jpg
│   ├── landmark81_002.jpg
│   └── ...
└── ben_thanh/
    ├── ben_thanh_001.jpg
    ├── ben_thanh_002.jpg
    └── ...
```

**Yêu cầu dữ liệu:**
- Định dạng: JPG, JPEG, PNG
- Kích thước tối thiểu: 224x224 pixels
- Khuyến nghị: 512x512 pixels trở lên
- Số lượng tối thiểu: 50 ảnh/địa điểm
- Khuyến nghị: 200+ ảnh/địa điểm

### 4. Huấn luyện Model

```bash
python train_landmarks.py
```

Quá trình huấn luyện sẽ:
- Tự động chia dữ liệu (80% training, 20% validation)
- Huấn luyện model ResNet50 với transfer learning
- Lưu model tốt nhất vào `landmark_models/best_model.pth`
- Tạo báo cáo đánh giá và biểu đồ

### 5. Test Hệ thống

```bash
python test_landmark_detection.py
```

## 📖 Sử dụng API

### Nhận diện Địa điểm

```python
from enhanced_landmark_detector import EnhancedLandmarkDetector

# Khởi tạo detector
detector = EnhancedLandmarkDetector()

# Nhận diện địa điểm
result = detector.detect_landmarks("path/to/image.jpg")

if result.get("landmark_detected"):
    landmark = result["primary_landmark"]
    print(f"Địa điểm: {landmark['name_vi']}")
    print(f"Vị trí: {landmark['location']}")
    print(f"Độ tin cậy: {landmark['confidence']:.2%}")
```

### Tích hợp với Image Analyzer

```python
from enhanced_image_analyzer import EnhancedImageAnalyzer

# Khởi tạo analyzer (đã tích hợp landmark detection)
analyzer = EnhancedImageAnalyzer()

# Phân tích hình ảnh
analysis = analyzer.analyze_image("path/to/image.jpg")

# Kết quả nhận diện địa điểm
landmark_result = analysis.get("landmark_detection", {})
```

### Huấn luyện Model Tùy chỉnh

```python
from landmark_detection_trainer import LandmarkDetectionTrainer

# Khởi tạo trainer
trainer = LandmarkDetectionTrainer()

# Chuẩn bị dữ liệu
train_loader, val_loader = trainer.prepare_data(image_paths, labels)

# Tạo và huấn luyện model
trainer.create_model()
best_accuracy = trainer.train(train_loader, val_loader)
```

## 🏛️ Địa điểm Được Hỗ trợ

| Địa điểm | Tên tiếng Việt | Tên tiếng Anh | Vị trí |
|----------|----------------|---------------|---------|
| bitexco | Tòa nhà Bitexco Financial Tower | Bitexco Financial Tower | District 1, HCMC |
| landmark81 | Tòa nhà Landmark 81 | Landmark 81 | Vinhomes Central Park |
| ben_thanh | Chợ Bến Thành | Ben Thanh Market | District 1, HCMC |
| notre_dame | Nhà thờ Đức Bà | Notre Dame Cathedral | District 1, HCMC |
| reunification_palace | Dinh Độc Lập | Reunification Palace | District 1, HCMC |
| war_remnants | Bảo tàng Chứng tích Chiến tranh | War Remnants Museum | District 3, HCMC |
| cu_chi_tunnels | Địa đạo Củ Chi | Cu Chi Tunnels | Cu Chi District |
| mekong_delta | Đồng bằng Sông Cửu Long | Mekong Delta | Southern Vietnam |
| phu_quoc | Đảo Phú Quốc | Phu Quoc Island | Kien Giang Province |
| ha_long_bay | Vịnh Hạ Long | Ha Long Bay | Quang Ninh Province |
| hoan_kiem | Hồ Hoàn Kiếm | Hoan Kiem Lake | Hanoi |
| temple_of_literature | Văn Miếu Quốc Tử Giám | Temple of Literature | Hanoi |

## 📊 Kết quả Mẫu

### Kết quả Nhận diện

```json
{
  "landmark_detected": true,
  "primary_landmark": {
    "key": "bitexco",
    "name_vi": "Tòa nhà Bitexco Financial Tower",
    "name_en": "Bitexco Financial Tower",
    "description_vi": "Tòa nhà chọc trời biểu tượng của TP.HCM",
    "location": "District 1, Ho Chi Minh City",
    "confidence": 0.89
  },
  "top_predictions": [
    {
      "landmark": "bitexco",
      "name": "Bitexco Financial Tower",
      "confidence": 0.89
    },
    {
      "landmark": "landmark81",
      "name": "Landmark 81",
      "confidence": 0.08
    },
    {
      "landmark": "other",
      "name": "Other/Unknown",
      "confidence": 0.03
    }
  ]
}
```

## 🔧 Tùy chỉnh

### Thêm Địa điểm Mới

1. Thêm thông tin địa điểm vào `vietnamese_landmarks` trong `enhanced_landmark_detector.py`
2. Tạo thư mục dữ liệu cho địa điểm mới
3. Huấn luyện lại model

### Điều chỉnh Model

- Thay đổi backbone: Sửa `models.resnet50` trong `LandmarkDetectionModel`
- Điều chỉnh hyperparameters: Sửa `batch_size`, `learning_rate`, `num_epochs`
- Thêm data augmentation: Sửa `train_transform`

## 📈 Đánh giá Hiệu suất

Sau khi huấn luyện, hệ thống sẽ tạo:

- **Báo cáo phân loại**: Precision, Recall, F1-score cho từng địa điểm
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Biểu đồ huấn luyện**: Loss và Accuracy theo epochs
- **Model tốt nhất**: Tự động lưu model có validation accuracy cao nhất

## 🐛 Xử lý Lỗi

### Lỗi thường gặp:

1. **Model không load được**
   - Kiểm tra file `landmark_models/best_model.pth` có tồn tại
   - Chạy lại `python train_landmarks.py`

2. **Không đủ dữ liệu huấn luyện**
   - Thêm ít nhất 50 ảnh cho mỗi địa điểm
   - Đảm bảo ảnh chất lượng tốt

3. **Lỗi CUDA/GPU**
   - Kiểm tra cài đặt PyTorch với CUDA
   - Hoặc sử dụng CPU: `torch.device("cpu")`

## 🤝 Đóng góp

Để cải thiện hệ thống:

1. Thêm dữ liệu huấn luyện chất lượng cao
2. Thử nghiệm các model backbone khác
3. Cải thiện data augmentation
4. Thêm địa điểm mới

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra logs trong console
2. Xem file `training_history.json` để debug
3. Chạy `python test_landmark_detection.py` để test

---

**Lưu ý**: Hệ thống này được thiết kế để hoạt động tốt nhất với dữ liệu huấn luyện đa dạng và chất lượng cao. Càng nhiều ảnh huấn luyện, độ chính xác càng cao.

