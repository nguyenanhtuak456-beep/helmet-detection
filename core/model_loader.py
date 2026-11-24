
from ultralytics import YOLO


def load_yolo_model(model_path):
    """
    Nạp mô hình YOLO từ đường dẫn file .pt.
    Việc cache model sẽ được quản lý ở app.py bằng Streamlit.
    """
    if not model_path:
        raise ValueError("Đường dẫn model chưa được cấu hình.")

    return YOLO(model_path)
