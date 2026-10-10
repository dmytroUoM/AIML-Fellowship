#!/usr/bin/env python3
"""
==================================================
Script: run_pipeline.py
Project 3: AIML-Driven Super-Resolution and Volumetric Reconstruction
           for Mixing Tanks

Purpose:
    Menu-driven entry point for the Project 3 image-processing pipeline.
    This thin wrapper validates required scripts, inputs, models and folders,
    then runs the existing Python scripts in the required order.

Menu:
    1. Image preprocessing: scripts 01, 02, 03 and 04
    2. TinyUNet inference: script 05
    3. LaMa inference: script 06
    4. Real-ESRGAN inference: script 07
    5. Run everything in order
    0. Exit

Usage:
    Place this file in Project3/Scripts, then run:
        python run_pipeline.py
    or on Windows:
        py run_pipeline.py
==================================================
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PYTHON_EXE = Path(sys.executable).resolve()

PREPROCESSING_SCRIPTS = (
    "01_CleanFolders.py",
    "02_Get_Avi_SHA256_Hash.py",
    "03_Get_Video_Metadata.py",
    "04_Get_Video_Frames.py",
)
TINYUNET_SCRIPT = "05_Run_TinyUNet_Inference.py"
LAMA_SCRIPT = "06_Run_LaMa_Inference.py"
REALESRGAN_SCRIPT = "07_Real-ESRGAN_Inference_Output.py"
ALL_SCRIPTS = (*PREPROCESSING_SCRIPTS, TINYUNET_SCRIPT, LAMA_SCRIPT, REALESRGAN_SCRIPT)

VIDEO_FILE = PROJECT_ROOT / "Video" / "active.avi"
FRAMES_DIR = PROJECT_ROOT / "Images" / "02_Frames"
LAMA_OUTPUT_DIR = PROJECT_ROOT / "Images" / "06_LaMa_Inference_Output"
TINYUNET_MODEL = PROJECT_ROOT / "Models" / "TinyUnet_Model.pt"
LAMA_MODEL = PROJECT_ROOT / "Models" / "big-lama.pt"
REALESRGAN_MODEL = PROJECT_ROOT / "Models" / "RealESRGAN_x4plus.pth"
LOGS_DIR = PROJECT_ROOT / "Logs"


class Colours:
    RESET = "\033[0m"
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"


def supports_colour() -> bool:
    return sys.stdout.isatty()


def colour(text: str, code: str) -> str:
    return f"{code}{text}{Colours.RESET}" if supports_colour() else text


def format_elapsed(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}m {remaining_seconds:02d}s"


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


def validate_path(path: Path, description: str, *, directory: bool = False) -> bool:
    valid = path.is_dir() if directory else path.is_file()
    if valid:
        return True
    expected_type = "folder" if directory else "file"
    print(colour(f"[ERROR] Missing {description} {expected_type}: {path}", Colours.RED))
    return False


def validate_scripts(script_names: Sequence[str]) -> bool:
    missing = [SCRIPT_DIR / name for name in script_names if not (SCRIPT_DIR / name).is_file()]
    if not missing:
        return True
    print(colour("[ERROR] Required pipeline scripts are missing:", Colours.RED))
    for path in missing:
        print(colour(f"  - {path}", Colours.RED))
    return False


def validate_installation() -> bool:
    """Confirm the wrapper is in Scripts and the expected project layout exists."""
    if SCRIPT_DIR.name.lower() != "scripts":
        print(colour("[ERROR] run_pipeline.py must be placed in the Project3/Scripts folder.", Colours.RED))
        print(colour(f"Current location: {SCRIPT_DIR}", Colours.RED))
        return False
    return validate_scripts(ALL_SCRIPTS)


def contains_png_files(folder: Path) -> bool:
    return folder.is_dir() and any(path.is_file() for path in folder.glob("*.png"))


def validate_png_input(folder: Path, description: str) -> bool:
    if not validate_path(folder, description, directory=True):
        return False
    if not contains_png_files(folder):
        print(colour(f"[ERROR] No PNG input files found in {description}: {folder}", Colours.RED))
        return False
    return True


def confirm_cleanup() -> bool:
    print(colour(
        "\nWARNING: 01_CleanFolders.py clears the contents of the Images and Reports folders.",
        Colours.YELLOW,
    ))
    return ask_yes_no("Continue with preprocessing?", default=False)


def child_environment() -> dict[str, str]:
    """Provide consistent project and log paths to environment-aware scripts."""
    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["LOG_DIR"] = str(LOGS_DIR)
    environment["LOGS_DIR"] = str(LOGS_DIR)
    return environment


def invoke_step(script_name: str, arguments: Sequence[str] = ()) -> bool:
    script_path = SCRIPT_DIR / script_name
    if not validate_path(script_path, "pipeline script"):
        return ask_yes_no("Continue with the remaining steps anyway?", default=False)

    command = [str(PYTHON_EXE), str(script_path), *map(str, arguments)]
    print()
    print(colour("=" * 60, Colours.CYAN))
    print(colour(f"Running: {script_name}", Colours.CYAN))
    if arguments:
        print(colour(f"Arguments: {' '.join(map(str, arguments))}", Colours.CYAN))
    print(colour(f"Working directory: {PROJECT_ROOT}", Colours.CYAN))
    print(colour("=" * 60, Colours.CYAN))

    step_start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment(),
            check=False,
        )
        exit_code = completed.returncode
    except KeyboardInterrupt:
        print(colour(f"\n[INTERRUPTED] {script_name}", Colours.RED))
        return False
    except OSError as exc:
        print(colour(f"[ERROR] Could not start {script_name}: {exc}", Colours.RED))
        exit_code = 1

    elapsed = format_elapsed(time.perf_counter() - step_start)
    if exit_code == 0:
        print(colour(f"[OK] {script_name} completed in {elapsed}", Colours.GREEN))
        return True

    print(colour(
        f"[FAILED] {script_name} exited with code {exit_code} after {elapsed}",
        Colours.RED,
    ))
    return ask_yes_no("Continue with the remaining steps anyway?", default=False)


def run_sequence(steps: Iterable[tuple[str, Sequence[str]]]) -> bool:
    for script_name, arguments in steps:
        if not invoke_step(script_name, arguments):
            print(colour("Sequence stopped.", Colours.RED))
            return False
    return True


def validate_preprocessing() -> bool:
    return validate_scripts(PREPROCESSING_SCRIPTS) and validate_path(VIDEO_FILE, "input video")


def validate_tinyunet() -> bool:
    return (
        validate_scripts((TINYUNET_SCRIPT,))
        and validate_path(TINYUNET_MODEL, "TinyUNet model")
        and validate_png_input(FRAMES_DIR, "TinyUNet input folder")
    )


def validate_lama() -> bool:
    return (
        validate_scripts((LAMA_SCRIPT,))
        and validate_path(LAMA_MODEL, "LaMa model")
        and validate_png_input(FRAMES_DIR, "LaMa input folder")
    )


def validate_realesrgan() -> bool:
    return (
        validate_scripts((REALESRGAN_SCRIPT,))
        and validate_path(REALESRGAN_MODEL, "Real-ESRGAN model")
        and validate_png_input(LAMA_OUTPUT_DIR, "Real-ESRGAN input folder")
    )


def validate_everything() -> bool:
    """Validate resources that must exist before the complete sequence starts."""
    return (
        validate_scripts(ALL_SCRIPTS)
        and validate_path(VIDEO_FILE, "input video")
        and validate_path(TINYUNET_MODEL, "TinyUNet model")
        and validate_path(LAMA_MODEL, "LaMa model")
        and validate_path(REALESRGAN_MODEL, "Real-ESRGAN model")
    )


def run_preprocessing() -> bool:
    print(colour("\n--- Image preprocessing: scripts 01 to 04 ---", Colours.MAGENTA))
    if not validate_preprocessing():
        return False
    if not confirm_cleanup():
        print(colour("Preprocessing cancelled.", Colours.YELLOW))
        return False
    return run_sequence((script_name, ()) for script_name in PREPROCESSING_SCRIPTS)


def run_tinyunet() -> bool:
    print(colour("\n--- TinyUNet inference ---", Colours.MAGENTA))
    return validate_tinyunet() and invoke_step(TINYUNET_SCRIPT)


def run_lama() -> bool:
    print(colour("\n--- LaMa inference ---", Colours.MAGENTA))
    return validate_lama() and invoke_step(LAMA_SCRIPT)


def run_realesrgan() -> bool:
    print(colour("\n--- Real-ESRGAN inference ---", Colours.MAGENTA))
    return validate_realesrgan() and invoke_step(REALESRGAN_SCRIPT)


def run_everything() -> bool:
    print(colour("\n--- Complete Project 3 pipeline ---", Colours.MAGENTA))
    if not validate_everything():
        return False
    if not confirm_cleanup():
        print(colour("Complete pipeline cancelled.", Colours.YELLOW))
        return False

    steps = [
        *((script_name, ()) for script_name in PREPROCESSING_SCRIPTS),
        (TINYUNET_SCRIPT, ()),
        (LAMA_SCRIPT, ()),
        (REALESRGAN_SCRIPT, ()),
    ]
    return run_sequence(steps)


def show_menu() -> None:
    print()
    print(colour("=" * 60, Colours.WHITE))
    print(colour(" Project 3 Pipeline - Run Menu", Colours.WHITE))
    print(colour("=" * 60, Colours.WHITE))
    print(" 1) Image preprocessing  (01-04: clean, hash, metadata, frames)")
    print(" 2) Run TinyUNet         (05)")
    print(" 3) Run LaMa             (06)")
    print(" 4) Run Real-ESRGAN      (07)")
    print(" 5) Run EVERYTHING       (01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07)")
    print(" 0) Exit")
    print(colour("=" * 60, Colours.WHITE))


def main() -> int:
    print(f"Python interpreter: {PYTHON_EXE}")
    print(f"Scripts folder: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")

    if not validate_installation():
        return 1

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    actions = {
        "1": run_preprocessing,
        "2": run_tinyunet,
        "3": run_lama,
        "4": run_realesrgan,
        "5": run_everything,
    }

    while True:
        show_menu()
        try:
            choice = input("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if choice == "0":
            print("\nDone.")
            return 0

        action = actions.get(choice)
        if action is None:
            print(colour("Not a valid option.", Colours.YELLOW))
            continue
        action()


if __name__ == "__main__":
    raise SystemExit(main())
