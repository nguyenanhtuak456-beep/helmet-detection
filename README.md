
# AI Traffic Vision

Hệ thống giám sát giao thông thông minh sử dụng YOLOv8 và ByteTrack.

## 1. Giới thiệu

AI Traffic Vision hỗ trợ phát hiện và thống kê người tham gia giao thông đội hoặc không đội mũ bảo hiểm thông qua video và hình ảnh.

## 2. Công nghệ sử dụng

- Python
- Streamlit
- YOLOv8
- ByteTrack
- PyTorch
- OpenCV
- SQLite

## 3. Chức năng

- Nhận diện mũ bảo hiểm.
- Nhận diện người không đội mũ bảo hiểm.
- Nhận diện xe máy.
- Theo dõi đối tượng bằng ByteTrack.
- Phân tích video và hình ảnh.
- Xử lý webcam.
- Thống kê kết quả nhận diện.
- Lưu và xem lại lịch sử phân tích.
- Quản lý cấu hình nhận diện.

## 4. Cấu trúc dự án

```text
AI_Traffic_Vision_Web/
├── app.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── model_config.py
│   ├── model_loader.py
│   ├── tracker_config.py
│   ├── video_utils.py
│   └── history_db.py
├── assets/
├── data/
│   └── videos/
└── .streamlit/
    └── config.toml
```

## 5. Cài đặt

Sử dụng Python 3.11.

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## 6. Mô hình AI

Đặt file trọng số YOLOv8 tại vị trí được cấu hình trong:

```text
core/model_config.py
```

Không đưa file best.pt lên GitHub công khai.

## 7. Chạy ứng dụng

```bash
streamlit run app.py
```

## 8. Lưu trữ dữ liệu

- SQLite lưu thông tin lịch sử phân tích.
- Video kết quả được lưu trong data/videos/.
- File model được quản lý riêng.

## 9. Triển khai

Dự án được chuẩn bị để triển khai trên máy chủ có môi trường Python và tài nguyên phù hợp cho xử lý AI.
