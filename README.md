# AI Traffic Vision — Interactive UI Demo

Bản giao diện tương tác để triển khai Streamlit Community Cloud mà không cần model, PyTorch, YOLOv8s, ByteTrack hoặc GPU.

## Tính năng
- Dashboard và số liệu minh họa.
- Tải video lên và phát video ngay trên giao diện.
- Điều chỉnh ngưỡng, bật/tắt lớp hiển thị và Track ID (chỉ là điều khiển UI).
- Tải và xem trước nhiều ảnh.
- Tạo phiên kết quả mô phỏng và xem lịch sử trong phiên chạy.
- Giao diện Dark Navy + Cyan.

**Lưu ý:** Không có nhận diện AI thật. Kết quả và biểu đồ là dữ liệu mô phỏng, được ghi chú rõ trong ứng dụng.

## Deploy Streamlit Community Cloud
1. Tạo GitHub repository.
2. Đưa `app.py`, `requirements.txt` và thư mục `.streamlit/` lên thư mục gốc repository.
3. Mở https://share.streamlit.io/ và chọn repository, branch, file `app.py`.
4. Deploy. Không cần tải `best.pt` hoặc cấu hình GPU.
