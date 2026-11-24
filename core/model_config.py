
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

    # 1. Kiểm tra đường dẫn model từ biến môi trường
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

    if local_path.is_file() and local_path.stat().st_size > 0:
        return str(local_path)

    # 3. Lấy mã file Google Drive từ Streamlit Secrets
    file_id = ""

    try:
        import streamlit as st
        file_id = st.secrets.get("GDRIVE_FILE_ID", "")
    except Exception:
        file_id = os.getenv("GDRIVE_FILE_ID", "")

    if not file_id:
        raise FileNotFoundError(
            "Chưa cấu hình GDRIVE_FILE_ID trong Streamlit Secrets."
        )

    # 4. Tạo thư mục lưu model
    model_dir = project_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best.pt"
    temp_path = model_dir / "best_download.tmp"

    # 5. Kiểm tra file model cũ nếu tồn tại
    if model_path.is_file():
        with open(model_path, "rb") as f:
            header = f.read(100)

        print("MODEL FILE SIZE:", model_path.stat().st_size)
        print("MODEL FILE HEADER:", repr(header))

        # Không sử dụng file rỗng hoặc file có nội dung văn bản lỗi
        if (
            model_path.stat().st_size > 0
            and not header.lstrip().startswith(
                (b"#", b"<", b"{", b"Access denied")
            )
        ):
            return str(model_path)

        print("File model cũ không hợp lệ, sẽ tải lại.")
        model_path.unlink()

    # 6. Xóa file tải tạm từ lần chạy trước
    if temp_path.exists():
        temp_path.unlink()

    # 7. Tải model từ Google Drive
    print("Đang tải YOLO model từ Google Drive...")

    try:
        downloaded_file = gdown.download(
            id=file_id,
            output=str(temp_path),
            quiet=False,
        )

        if (
            downloaded_file is None
            or not temp_path.is_file()
            or temp_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Google Drive không trả về file model hợp lệ."
            )

        # Kiểm tra nội dung tải về
        with open(temp_path, "rb") as f:
            header = f.read(100)

        print("DOWNLOADED MODEL SIZE:", temp_path.stat().st_size)
        print("DOWNLOADED MODEL HEADER:", repr(header))

        if header.lstrip().startswith(
            (b"#", b"<", b"{", b"Access denied")
        ):
            raise RuntimeError(
                "Google Drive trả về nội dung văn bản thay vì file best.pt. "
                "Hãy kiểm tra quyền chia sẻ và liên kết Google Drive."
            )

        # Chuyển file tải thành best.pt chính thức
        os.replace(temp_path, model_path)

        print("Đã tải YOLO model thành công.")

        return str(model_path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
