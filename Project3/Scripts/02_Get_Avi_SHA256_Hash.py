#!/usr/bin/env python3
# ============================================
# Script: 02_Get_Avi_SHA256_Hash.py
#
# Project 2:
# AIML-Driven Super-Resolution and
# Volumetric Reconstruction for Mixing Tanks
#
# Purpose:
#   - Calculate SHA256 hash of active.avi
#   - Save hash to Reports folder
#   - Save log to Logs folder beside script
# ============================================
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path



parser = argparse.ArgumentParser()
parser.add_argument(
    "--video-file-name",
    default="active.avi"
)

args = parser.parse_args()


def write_log(
    message: str,
    level: str = "INFO",
    log_file: Path | None = None,
):
    """
    Write log message to console and log file.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"

    print(log_entry)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest().upper()


def main():

    # Script directory
    script_dir = Path(__file__).resolve().parent

    # Project root is parent of Scripts folder
    project_root = Path.cwd()

    # Paths
    video_file = project_root / "Video" / "active.avi"

    reports_dir = project_root / "Reports"
    logs_dir = Path("/workspace/Logs")

    hash_file = reports_dir / "02_Avi_SHA256_Hash.txt"
    log_file = logs_dir / "02_Get_Avi_SHA256_Hash.log"

    # Ensure folders exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:

        write_log("============================================", log_file=log_file)
        write_log("Script started.", log_file=log_file)

        write_log(
            f"Script folder: {script_dir}",
            log_file=log_file,
        )

        write_log(
            f"Project root: {project_root}",
            log_file=log_file,
        )

        write_log(
            f"Video file: {video_file}",
            log_file=log_file,
        )

        write_log(
            f"Reports folder: {reports_dir}",
            log_file=log_file,
        )

        write_log(
            f"Logs folder: {logs_dir}",
            log_file=log_file,
        )

        # Verify AVI exists
        if not video_file.is_file():
            raise FileNotFoundError(
                f"Video file not found: {video_file}"
            )

        write_log(
            "Calculating SHA256 hash.",
            log_file=log_file,
        )

        # Calculate hash
        hash_value = calculate_sha256(video_file)

        # Save hash
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(hash_value)

        write_log(
            "SHA256 hash calculated successfully.",
            log_file=log_file,
        )

        write_log(
            f"Hash saved to: {hash_file}",
            log_file=log_file,
        )

        write_log(
            f"Hash value: {hash_value}",
            log_file=log_file,
        )

        write_log(
            "Script completed successfully.",
            level="SUCCESS",
            log_file=log_file,
        )

        write_log("============================================", log_file=log_file)

    except Exception as e:

        write_log(
            f"Script failed: {e}",
            level="ERROR",
            log_file=log_file,
        )

        write_log("============================================", log_file=log_file)

        sys.exit(1)


if __name__ == "__main__":
    main()