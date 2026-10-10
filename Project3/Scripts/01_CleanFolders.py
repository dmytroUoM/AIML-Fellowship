#!/usr/bin/env python3
"""
==================================================
Script: 01_CleanFolders.py

PURPOSE
    Safely cleans the contents of approved output folders.

DESCRIPTION
    Removes all files and subfolders inside predefined output
    folders while preserving the folders themselves.

    Approved folders:
    - ../Images
    - ../Reports

EXAMPLES
    python 01_CleanFolders.py
    python 01_CleanFolders.py --what-if
==================================================
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


def write_log(
    message: str,
    level: str = "INFO",
    log_file: Path | None = None,
):
    """Write message to console and log file."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} [{level}] {message}"

    print(entry)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safely clean approved output folders."
    )

    parser.add_argument(
        "--what-if",
        action="store_true",
        help="Simulate cleanup without deleting files.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    script_root = Path(__file__).resolve().parent

    folders = [
        script_root.parent / "Images",
        script_root.parent / "Reports",
    ]

    approved_folder_names = {
        "Images",
        "Reports",
    }

    log_folder = script_root / "Logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file = log_folder / "01_CleanFolders.log"

    write_log(
        "--------------------------------------------------",
        log_file=log_file,
    )

    write_log(
        "Starting folder cleanup script.",
        log_file=log_file,
    )

    write_log(
        f"Script location: {script_root}",
        log_file=log_file,
    )

    if args.what_if:
        write_log(
            "Running in WHAT-IF mode. No files will be deleted.",
            level="WARNING",
            log_file=log_file,
        )

    for folder in folders:

        try:

            write_log(
                f"Processing folder reference: {folder}",
                log_file=log_file,
            )

            if not folder.exists():

                write_log(
                    f"Folder not found: {folder}",
                    level="WARNING",
                    log_file=log_file,
                )

                continue

            full_path = folder.resolve()
            folder_name = full_path.name

            write_log(
                f"Resolved path: {full_path}",
                log_file=log_file,
            )

            # Safety check 1
            if not str(full_path).strip():

                write_log(
                    "Resolved folder path is empty or invalid.",
                    level="ERROR",
                    log_file=log_file,
                )

                continue

            # Safety check 2
            if folder_name not in approved_folder_names:

                write_log(
                    f"Folder is not approved: {full_path}",
                    level="ERROR",
                    log_file=log_file,
                )

                continue

            # Safety check 3
            if full_path == full_path.anchor:

                write_log(
                    f"Refusing to clean root path: {full_path}",
                    level="ERROR",
                    log_file=log_file,
                )

                continue

            items = list(full_path.iterdir())

            if not items:

                write_log(
                    f"Folder already empty: {full_path}",
                    log_file=log_file,
                )

                continue

            write_log(
                f"Items found for deletion in {full_path}: {len(items)}",
                log_file=log_file,
            )

            for item in items:

                if args.what_if:

                    write_log(
                        f"WHAT-IF: Would delete {item}",
                        level="WARNING",
                        log_file=log_file,
                    )

                    continue

                write_log(
                    f"Deleting: {item}",
                    log_file=log_file,
                )

                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            if args.what_if:

                write_log(
                    f"WHAT-IF: Cleanup simulated for folder: {full_path}",
                    level="WARNING",
                    log_file=log_file,
                )

            else:

                write_log(
                    f"Successfully cleaned folder: {full_path}",
                    log_file=log_file,
                )

        except Exception as ex:

            write_log(
                f"Failed to clean folder '{folder}'. Error: {ex}",
                level="ERROR",
                log_file=log_file,
            )

    write_log(
        "Folder cleanup script completed.",
        log_file=log_file,
    )

    write_log(
        f"Log file location: {log_file}",
        log_file=log_file,
    )

    write_log(
        "--------------------------------------------------",
        log_file=log_file,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)