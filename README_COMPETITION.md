# 🚀 AI Challenge V - Competition Ready System

Hệ thống tìm kiếm hình ảnh thông minh được tối ưu hóa cho cuộc thi AI Challenge V với khả năng xử lý dữ liệu 100GB+ và độ chính xác cao.

## 🎯 Tính năng chính

### 🔍 Tìm kiếm thông minh
- **Text-to-Image Search**: Tìm kiếm hình ảnh bằng câu truy vấn tiếng Việt/Anh
- **Image-to-Image Search**: Tìm kiếm hình ảnh tương tự
- **Translation**: Tự động dịch truy vấn tiếng Việt sang tiếng Anh
- **Similarity Ranking**: Sắp xếp kết quả theo độ tương đồng

### 📊 Xử lý dữ liệu lớn
- **100GB+ Data Support**: Xử lý hiệu quả với dữ liệu khổng lồ
- **Real-time Processing**: Xử lý dữ liệu mới liên tục từ thư mục D:/images
- **Smart Caching**: Cache thông minh để tăng tốc độ truy vấn
- **Batch Processing**: Xử lý theo batch để tối ưu bộ nhớ

### 🎨 Giao diện tối ưu
- **Pure JavaScript**: Frontend nhanh với JavaScript thuần
- **Responsive Design**: Tương thích mọi thiết bị
- **Image Modal**: Xem chi tiết hình ảnh với thông tin đầy đủ
- **Query Suggestions**: Gợi ý truy vấn thông minh
- **Performance Indicators**: Hiển thị độ chính xác và xếp hạng

### ⚡ Hiệu năng cao
- **15-second Query Limit**: Tối ưu cho giới hạn thời gian cuộc thi
- **FAISS Index**: Tìm kiếm vector nhanh chóng
- **CLIP Model**: Mô hình AI tiên tiến cho tìm kiếm đa phương thức
- **Memory Optimization**: Quản lý bộ nhớ hiệu quả

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- RAM: 8GB+ (khuyến nghị 16GB+)
- Disk: 50GB+ free space
- Windows/Linux/macOS

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd Pipeline_HCM_AI
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu
- Đặt dữ liệu hình ảnh vào thư mục `D:/images`
- Hệ thống sẽ tự động xử lý và index dữ liệu

### Bước 4: Khởi chạy hệ thống
```bash
python competition_launcher.py
```

## 🚀 Sử dụng

### Khởi động nhanh
```bash
# Chạy launcher (khuyến nghị)
python competition_launcher.py

# Hoặc chạy từng component
python data_processor.py  # Terminal 1
python app.py            # Terminal 2
```

### Truy cập giao diện
- **Search Interface**: http://localhost:5001
- **API Status**: http://localhost:5001/api/status
- **API Search**: http://localhost:5001/api/search?q=query

### Sử dụng API
```bash
# Text search
curl "http://localhost:5001/api/search?q=person%20walking&type=text&k=50"

# Get images
curl "http://localhost:5001/api/images?page=1&per_page=100"

# Get status
curl "http://localhost:5001/api/status"
```

## 📁 Cấu trúc dự án

```
Pipeline_HCM_AI/
├── app.py                          # Flask web application
├── data_processor.py               # Data processing engine
├── competition_launcher.py         # System launcher
├── requirements.txt                # Python dependencies
├── competition_config.json         # Competition settings
├── templates/
│   └── home_optimized_v2.html     # Main UI template
├── static/
│   ├── css/
│   ├── js/
│   └── thumbnails/                 # Generated thumbnails
├── cache/
│   ├── processed_files.db          # SQLite database
│   ├── faiss_index.bin            # FAISS search index
│   ├── image_path.json            # Image paths mapping
│   └── competition_metadata.json  # Image metadata
└── logs/
    ├── data_processor.log         # Processing logs
    └── competition_launcher.log   # Launcher logs
```

## 🔧 Cấu hình

### Competition Configuration
File `competition_config.json`:
```json
{
  "competition_mode": true,
  "max_query_time": 15,
  "max_results": 100,
  "cache_timeout": 300,
  "batch_size": 128,
  "max_workers": 8,
  "health_check_interval": 30,
  "performance_monitoring": true,
  "auto_recovery": true
}
```

### Environment Variables
```bash
# Data source directory
export DATA_SOURCE_DIR="D:/images"

# Cache directory
export CACHE_DIR="./cache"

# Flask port
export FLASK_PORT=5001

# Log level
export LOG_LEVEL=INFO
```

## 📊 Monitoring & Performance

### Health Checks
- **Data Processor**: Kiểm tra database và xử lý file
- **Flask App**: Kiểm tra API endpoints và system status
- **Auto Recovery**: Tự động khởi động lại nếu component lỗi

### Performance Metrics
- **Query Response Time**: < 15 seconds
- **Memory Usage**: < 80% RAM
- **CPU Usage**: < 90%
- **Disk Usage**: < 95%

### Logs
```bash
# View real-time logs
tail -f data_processor.log
tail -f competition_launcher.log

# Check system status
curl http://localhost:5001/api/status
```

## 🐛 Troubleshooting

### Lỗi thường gặp

#### 1. "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. "Database locked"
```bash
# Restart data processor
pkill -f data_processor.py
python data_processor.py
```

#### 3. "Out of memory"
```bash
# Reduce batch size in competition_config.json
"batch_size": 64
```

#### 4. "FAISS index not found"
```bash
# Wait for data processing to complete
# Check cache/faiss_index.bin exists
```

### Performance Issues

#### Slow Search
- Tăng `max_workers` trong config
- Giảm `batch_size` nếu memory thấp
- Kiểm tra FAISS index size

#### High Memory Usage
- Giảm `batch_size`
- Tăng `cache_timeout`
- Restart system

#### Slow UI
- Kiểm tra network connection
- Giảm `per_page` trong API calls
- Clear browser cache

## 🎯 Competition Tips

### Tối ưu truy vấn
1. **Sử dụng từ khóa cụ thể**: "person walking" thay vì "people"
2. **Kết hợp từ khóa**: "red car driving fast"
3. **Sử dụng gợi ý**: Click vào suggestion tags

### Xử lý kết quả
1. **Kiểm tra độ chính xác**: Màu viền xanh = cao, vàng = trung bình, đỏ = thấp
2. **Xem chi tiết**: Click vào ảnh để xem thông tin đầy đủ
3. **Ghi tên file**: Copy tên file để nhập đáp án

### Performance trong thi
1. **Preload data**: Đảm bảo dữ liệu đã được xử lý trước
2. **Monitor resources**: Theo dõi CPU/Memory usage
3. **Backup plan**: Có sẵn plan B nếu hệ thống lỗi

## 🔄 Updates & Maintenance

### Cập nhật dữ liệu
```bash
# Dữ liệu mới sẽ tự động được xử lý
# Hoặc restart data processor
python data_processor.py
```

### Backup
```bash
# Backup cache directory
cp -r cache/ cache_backup_$(date +%Y%m%d_%H%M%S)/
```

### Cleanup
```bash
# Clear old logs
rm -f *.log

# Clear cache (careful!)
rm -rf cache/*
```

## 📞 Support

### Logs Location
- `data_processor.log`: Data processing logs
- `competition_launcher.log`: System launcher logs
- `app.log`: Flask application logs

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python competition_launcher.py
```

### Emergency Restart
```bash
# Kill all processes
pkill -f "python.*(app|data_processor|competition_launcher)"

# Restart launcher
python competition_launcher.py
```

## 🏆 Competition Checklist

### Trước thi
- [ ] Hệ thống chạy ổn định
- [ ] Dữ liệu đã được xử lý hoàn toàn
- [ ] API endpoints hoạt động
- [ ] UI responsive và nhanh
- [ ] Backup dữ liệu

### Trong thi
- [ ] Monitor system performance
- [ ] Ghi tên file chính xác
- [ ] Sử dụng query suggestions
- [ ] Kiểm tra độ chính xác kết quả

### Sau thi
- [ ] Backup logs và kết quả
- [ ] Document performance metrics
- [ ] Analyze query patterns

---

**🎉 Chúc bạn thành công trong cuộc thi AI Challenge V! 🎉**



