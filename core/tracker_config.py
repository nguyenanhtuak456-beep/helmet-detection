
import os
import tempfile


def get_custom_tracker_yaml():
    """
    Tạo file cấu hình ByteTrack tùy chỉnh.
    Giữ nguyên các tham số từ phiên bản AI Traffic Vision gốc.
    """
    tracker_path = os.path.join(
        tempfile.gettempdir(),
        "bytetrack_custom.yaml"
    )

    tracker_config = (
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.10\n"
        "new_track_thresh: 0.30\n"
        "track_buffer: 90\n"
        "match_thresh: 0.65\n"
        "fuse_score: True\n"
    )

    with open(tracker_path, "w", encoding="utf-8") as f:
        f.write(tracker_config)

    return tracker_path
