# ============================================
# Script: 04_Get_Video_Frames.py
#
# Purpose:
#   - Extract frames from active.avi using OpenCV
#   - Save original frames to Images/01_Origin
#   - Save cropped frames to Images/02_Frames
#   - Crop: y=40:640, x=0:940
#   - Create output folders automatically
#   - Save execution log
# ============================================

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and crop frames from an AVI video using OpenCV."
    )

    parser.add_argument(
        "--video-file-name",
        type=str,
        default="active.avi",
        help="Video file name inside the Video folder",
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file creation",
    )

    return parser.parse_args()


def write_log(
    message: str,
    level: str = "INFO",
    log_file: Path = None,
    enable_logging: bool = True,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"[{timestamp}] [{level}] {message}"

    print(entry)

    if enable_logging and log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def main():
    args = parse_args()

    enable_logging = not args.no_log

    script_dir = Path(__file__).resolve().parent
    project_root = Path.cwd()

    avi_file = project_root / "Video" / args.video_file_name

    origin_dir = project_root / "Images" / "01_Origin"
    frames_dir = project_root / "Images" / "02_Frames"

    logs_dir = Path("/workspace/Logs")
    log_file = logs_dir / "04_Get_Video_Frames.log"

    # Create folders
    origin_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if enable_logging:
        logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        write_log("============================================",
                  log_file=log_file,
                  enable_logging=enable_logging)

        write_log("Script started.",
                  log_file=log_file,
                  enable_logging=enable_logging)

        write_log(f"Script folder: {script_dir}",
                  log_file=log_file,
                  enable_logging=enable_logging)

        write_log(f"Project root: {project_root}",
                  log_file=log_file,
                  enable_logging=enable_logging)

        write_log(f"Input video file: {avi_file}",
                  log_file=log_file,
                  enable_logging=enable_logging)

        write_log(f"Original frames folder: {origin_dir}",
                  log_file=log_file,
                  enable_logging=enable_logging)

        write_log(f"Cropped frames folder: {frames_dir}",
                  log_file=log_file,
                  enable_logging=enable_logging)

        if not avi_file.is_file():
            raise FileNotFoundError(
                f"Input video file not found: {avi_file}"
            )

        video = cv2.VideoCapture(str(avi_file))

        if not video.isOpened():
            raise RuntimeError(
                f"Cannot open video: {avi_file}"
            )

        frame_count = 0

        write_log(
            "Starting frame extraction...",
            log_file=log_file,
            enable_logging=enable_logging,
        )

        while True:

            success, frame = video.read()

            if not success:
                break

            frame_count += 1

            # Save original frame
            original_file = origin_dir / f"frame_{frame_count:04d}.png"

            cv2.imwrite(
                str(original_file),
                frame
            )

            # Crop frame
            cropped = frame[40:640, 0:940]

            # Save cropped frame
            cropped_file = frames_dir / f"frame_{frame_count:04d}.png"

            cv2.imwrite(
                str(cropped_file),
                cropped
            )

        video.release()

        total_original = len(list(origin_dir.glob("frame_*.png")))
        total_cropped = len(list(frames_dir.glob("frame_*.png")))

        write_log(
            f"Original frames saved: {total_original}",
            level="SUCCESS",
            log_file=log_file,
            enable_logging=enable_logging,
        )

        write_log(
            f"Cropped frames saved: {total_cropped}",
            level="SUCCESS",
            log_file=log_file,
            enable_logging=enable_logging,
        )

        write_log(
            "Script completed successfully.",
            level="SUCCESS",
            log_file=log_file,
            enable_logging=enable_logging,
        )

        write_log("============================================",
                  log_file=log_file,
                  enable_logging=enable_logging)

    except Exception as e:

        write_log(
            f"Script failed: {e}",
            level="ERROR",
            log_file=log_file,
            enable_logging=enable_logging,
        )

        write_log("============================================",
                  log_file=log_file,
                  enable_logging=enable_logging)

        sys.exit(1)


if __name__ == "__main__":
    main()