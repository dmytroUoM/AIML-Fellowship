#!/usr/bin/env python3
# ============================================
# Script: 03_Get_Video_Metadata.py
#
# Project 2:
# AIML-Driven Super-Resolution and
# Volumetric Reconstruction for Mixing Tanks
#
# Purpose:
#   - Get video metadata using OpenCV
#   - Save JSON report to Reports folder
#   - Save log to Logs folder
#   - Create Reports and Logs folders automatically
# ============================================

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video metadata using OpenCV."
    )

    parser.add_argument(
        "--video-file-name",
        default="active.avi",
        help="Video file located in Video folder."
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log file creation."
    )

    return parser.parse_args()


def write_log(
    message: str,
    level: str = "INFO",
    log_file: Path | None = None,
    enable_logging: bool = True
):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    entry = f"[{timestamp}] [{level}] {message}"

    print(entry)

    if enable_logging and log_file:
        with open(
            log_file,
            "a",
            encoding="utf-8"
        ) as f:
            f.write(entry + "\n")


def fourcc_to_string(fourcc: int) -> str:
    """Convert OpenCV FOURCC integer to codec string."""

    return "".join(
        chr((fourcc >> (8 * i)) & 0xFF)
        for i in range(4)
    ).strip()


def main():

    args = parse_args()

    enable_logging = not args.no_log

    script_dir = Path(__file__).resolve().parent
    project_root = Path.cwd()

    video_file = (
        project_root
        / "Video"
        / args.video_file_name
    )

    reports_dir = project_root / "Reports"
    logs_dir = Path("/workspace/Logs")

    output_file = (
        reports_dir
        / "03_Video_Metadata.json"
    )

    log_file = (
        logs_dir
        / "03_Get_Video_Metadata.log"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if enable_logging:
        logs_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    try:

        write_log(
            "============================================",
            log_file=log_file,
            enable_logging=enable_logging
        )

        write_log(
            "Script started.",
            log_file=log_file,
            enable_logging=enable_logging
        )

        write_log(
            f"Script name: {Path(__file__).name}",
            log_file=log_file,
            enable_logging=enable_logging
        )

        write_log(
            f"Script folder: {script_dir}",
            log_file=log_file,
            enable_logging=enable_logging
        )

        write_log(
            f"Project root: {project_root}",
            log_file=log_file,
            enable_logging=enable_logging
        )

        write_log(
            f"Video file: {video_file}",
            log_file=log_file,
            enable_logging=enable_logging
        )

        if not video_file.is_file():
            raise FileNotFoundError(
                f"Video file not found: {video_file}"
            )

        file_stat = video_file.stat()

        write_log(
            f"Video file size: {file_stat.st_size} bytes",
            log_file=log_file,
            enable_logging=enable_logging
        )

        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_file}"
            )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fourcc = int(
            cap.get(
                cv2.CAP_PROP_FOURCC
            )
        )

        codec = fourcc_to_string(
            fourcc
        )

        duration_seconds = (
            frame_count / fps
            if fps > 0
            else 0
        )

        cap.release()

        metadata = {
            "file_name": video_file.name,
            "size_bytes": file_stat.st_size,
            "modified": datetime.fromtimestamp(
                file_stat.st_mtime
            ).isoformat(),
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_seconds": round(
                duration_seconds,
                3
            ),
            "codec": codec
        }

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=4
            )

        write_log(
            "Metadata extracted successfully.",
            "SUCCESS",
            log_file,
            enable_logging
        )

        write_log(
            f"Metadata saved to: {output_file}",
            log_file=log_file,
            enable_logging=enable_logging
        )

        write_log(
            "Script completed successfully.",
            "SUCCESS",
            log_file,
            enable_logging
        )

        write_log(
            "============================================",
            log_file=log_file,
            enable_logging=enable_logging
        )

    except Exception as ex:

        write_log(
            f"Script failed: {ex}",
            "ERROR",
            log_file,
            enable_logging
        )

        write_log(
            "============================================",
            log_file=log_file,
            enable_logging=enable_logging
        )

        sys.exit(1)


if __name__ == "__main__":
    main()