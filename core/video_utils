
import os
import shutil
import subprocess


def convert_to_browser_mp4(input_path, output_path):
    """Chuyển video sang chuẩn H.264 + faststart để phát trên trình duyệt."""

    ffmpeg_bin = shutil.which("ffmpeg")

    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"

    subprocess.run(
        [
            ffmpeg_bin, "-y", "-i", input_path,
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-crf", "26", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return (
        os.path.exists(output_path)
        and os.path.getsize(output_path) > 1024
    )
