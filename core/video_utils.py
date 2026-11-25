# -*- coding: utf-8 -*-
"""Tiện ích video. Thiếu ffmpeg thì trả False để app dùng file mp4 gốc, không bị lỗi."""
import os
import shutil
import subprocess


def _find_ffmpeg():
    """Ưu tiên ffmpeg của hệ thống, sau đó dùng bản đóng gói sẵn của imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_browser_mp4(input_path, output_path):
    """Đổi sang H.264 để trình duyệt phát được. Trả True nếu thành công, False nếu không."""
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", "-an",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return (
        result.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 0
    )
