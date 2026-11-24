
import os
from pathlib import Path


def get_model_path():
    """
    Lấy đường dẫn model YOLO từ biến môi trường.
    Nếu chưa cấu hình, thử tìm best.pt trong thư mục dự án.
    """

    # Ưu tiên đường dẫn do máy chủ cung cấp
    env_path = os.getenv("AI_TRAFFIC_MODEL_PATH")

    if env_path:
        model_path = Path(env_path).expanduser()

        if model_path.is_file():
            return str(model_path)

        raise FileNotFoundError(
            f"Không tìm thấy model tại: {model_path}"
        )

    # Dùng best.pt trong thư mục dự án khi chạy thử local
    local_path = Path(__file__).resolve().parent.parent / "best.pt"

    if local_path.is_file():
        return str(local_path)

    raise FileNotFoundError(
        "Không tìm thấy best.pt. "
        "Hãy cấu hình AI_TRAFFIC_MODEL_PATH."
    )
