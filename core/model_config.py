
import os
from pathlib import Path

import gdown


def get_model_path():
    """
    Lấy đường dẫn YOLO model theo thứ tự:
    1. Biến môi trường AI_TRAFFIC_MODEL_PATH.
    2. File best.pt có sẵn trong dự án.
    3. Tải best.pt từ Google Drive.
    """

    # 1. Kiểm tra đường dẫn model được cấu hình bằng biến môi trường
    env_path = os.getenv("AI_TRAFFIC_MODEL_PATH")

    if env_path:
        model_path = Path(env_path).expanduser()

        if model_path.is_file():
            return str(model_path)

        raise FileNotFoundError(
            f"Không tìm thấy model tại: {model_path}"
        )

    # 2. Kiểm tra model có sẵn trong thư mục dự án
    project_path = Path(__file__).resolve().parent.parent
    local_path = project_path / "best.pt"

    if local_path.is_file():
        return str(local_path)

    # 3. Lấy mã file Google Drive từ Streamlit Secrets
    try:
        import streamlit as st
        file_id = st.secrets.get("GDRIVE_FILE_ID", "")
    except Exception:
        file_id = os.getenv("GDRIVE_FILE_ID", "")

    if not file_id:
        raise FileNotFoundError(
            "Chưa cấu hình GDRIVE_FILE_ID trong Streamlit Secrets."
        )

    # Tạo thư mục lưu model
    model_dir = project_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best.pt"

    # Kiểm tra file model đã tồn tại
if model_path.is_file():
    with open(model_path, "rb") as f:
        header = f.read(100)

    print("MODEL FILE SIZE:", model_path.stat().st_size)
    print("MODEL FILE HEADER:", repr(header))

    # Tạm thời xóa file cũ để tải lại
    model_path.unlink()

    # Tải model từ Google Drive
    print("Đang tải YOLO model từ Google Drive...")

    downloaded_file = gdown.download(
    id=file_id,
    output=str(model_path),
    quiet=False,
)

    if (
        downloaded_file is None
        or not model_path.is_file()
        or model_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Không tải được best.pt từ Google Drive. "
            "Hãy kiểm tra quyền chia sẻ của file."
        )

    print("Đã tải YOLO model thành công.")

    return str(model_path)
