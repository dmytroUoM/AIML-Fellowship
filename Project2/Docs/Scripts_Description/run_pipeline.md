# Script Description: run_pipeline.ps1

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `run_pipeline.ps1` |
| **Language** | PowerShell |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Pipeline Orchestration / Workflow Automation / CLI Menu |
| **Platform** | Windows PowerShell (or PowerShell Core, cross-platform) |
| **Execution Type** | Interactive menu-driven script; run manually (`.\run_pipeline.ps1`), no arguments required |

---

## Categories

1. **Pipeline Orchestration**
   Provides a single, menu-driven entry point that runs the project's existing `.ps1`/`.py` scripts in the correct order, without duplicating any of their internal logic.

2. **Workflow Branch Management**
   Offers separate execution paths for the legacy core pipeline, the legacy visual (classical) pipeline, and the AIML data/training branch, so users can run only the portion of the pipeline relevant to their current task.

3. **Failure Handling & Operator Control**
   Detects non-zero exit codes from each step and pauses to ask whether to continue, rather than silently propagating a failure through the rest of a sequence.

---

## Script Purpose

`run_pipeline.ps1` is a thin orchestration wrapper that lets a user run any stage — or the entirety — of Project 2's processing pipeline from a single interactive menu, without needing to remember each script's name, argument set, or execution order. It calls the existing scripts with their existing parameters; each underlying script's own log file remains the authoritative record of what happened during that step.

### Key Features & Execution Flow

* **Python Launcher Detection (`Get-PythonLauncher`):**
  * Prefers the `py` launcher (matching this project's convention), falling back to `python` if `py` is not on `PATH`; throws a clear error if neither is found.
* **Generic Step Runner (`Invoke-Step`):**
  * Runs a named `.ps1` or `.py` script with optional arguments, times its execution, reports pass/fail with colour-coded console output, and — on failure — asks the user whether to continue with the rest of the sequence.
  * Skips (with a warning) any script file that cannot be found at the expected path, rather than crashing the whole menu.
* **Five Composable Sequences:**
  * `Run-LegacyCore` — steps `00`–`05`: raw video to cropped frames.
  * `Run-LegacyVisual` — steps `06`–`10`: classical cleanup with an optional gradient-smoothing step before Real-ESRGAN upscaling.
  * `Run-AimlDataBranch` — steps `aiml_06`–`aiml_08` (split preparation): AI-assisted cleanup, synthetic background generation, and dataset splitting.
  * `Run-Training` — runs `aiml_08_train_segmentation_demo.py` on CPU directly, or points the user to `aiml_08_submit_gpu_training.sh` for GPU/HPC training.
  * `Run-Inference` — runs `aiml_09_run_trained_model_transparent.py` on either the scored validation set or new unlabeled frames.
* **Interactive Menu (`Show-Menu`):**
  * Presents six numbered options (the five sequences above, plus "Run EVERYTHING in order") and an exit option, looping until the user chooses to exit.
* **"Run Everything" Composition:**
  * Option 6 runs the legacy core sequence first, then prompts the user to choose the legacy visual branch, the AIML branch (which chains data prep → training → inference), or both.

### Step-by-Step Process Workflow

1. **Environment Setup:**
   * Sets `$ErrorActionPreference = "Stop"`, resolves the script's own directory, and detects an available Python launcher.

2. **Menu Loop:**
   * Displays the menu and reads the user's choice repeatedly until `0` (Exit) is selected.

3. **Sequence Dispatch:**
   * Routes the chosen option to the corresponding `Run-*` function, each of which calls `Invoke-Step` for its constituent scripts in order, stopping the sequence early if a step fails and the user declines to continue.

4. **Step Execution & Reporting:**
   * For each step: prints a banner, executes the script (with any required arguments), times it, and prints a colour-coded `[OK]` or `[FAILED]` summary.

5. **Exit:**
   * Prints "Done." once the user selects option `0`.

---

## Business / Process Purpose

Running each stage of a multi-script video-to-model pipeline by hand is error-prone and easy to get out of order, especially across a project with both a legacy classical branch and an AIML data/training branch.

`run_pipeline.ps1` supports the project by:
1. **Reducing Operator Error:** Encoding the correct script order and required arguments once, in one place, rather than relying on the user to remember or re-type them each session.
2. **Supporting Partial and Full Runs:** Letting a user re-run just one branch (e.g. only the AIML data branch after tweaking synthetic background generation) without needing to run the entire pipeline from scratch.
3. **Providing a Foundation for a Future Front-End:** Keeping each pipeline stage as a separate, single-purpose function (`Run-LegacyCore`, `Run-LegacyVisual`, `Run-AimlDataBranch`, `Run-Training`, `Run-Inference`) so the same logic can later be reused by a simple web or desktop UI, as noted in the script's own closing comments.

---

## Command-Line Arguments & Parameters

`run_pipeline.ps1` takes no command-line arguments; all choices are made interactively via the on-screen menu and follow-up prompts.

| Menu Option | Action |
| :---: | :--- |
| `1` | Run legacy core (steps `00`–`05`): video → cropped frames. |
| `2` | Run legacy visual pipeline (steps `06`–`10`): classical cleanup + optional AI upscale. |
| `3` | Run AIML data branch (`aiml_06`–`aiml_08` split preparation). |
| `4` | Train segmentation model (`aiml_08_train_segmentation_demo.py`, CPU only from this menu). |
| `5` | Run inference (`aiml_09_run_trained_model_transparent.py`) on validation or new frames. |
| `6` | Run everything in order (`1` → `2` and/or `3` → `4` → `5`, with branch selection). |
| `0` | Exit the menu loop. |

Interactive sub-prompts within these options include: whether to run gradient smoothing before Real-ESRGAN (option `2`), whether to train on CPU or GPU (option `4`; GPU training redirects the user to submit `aiml_08_submit_gpu_training.sh` to a SLURM cluster rather than running locally), and whether to run inference on the scored validation set or new unlabeled frames (option `5`).

---

## Orchestrated Scripts by Sequence

```text
Run-LegacyCore:
  00_CleanFolders.ps1
  01_get_avi_hash.ps1
  02_get_metadata_ffprob.ps1
  03_get_avi_metadata_powershell.ps1
  04_extract_frames_from_avi.ps1
  05_extract_crop_frames_from_avi.ps1

Run-LegacyVisual:
  06_artifacts_removal.py
  07_make_transparent_background.py
  08_round_edges.py
  [optional] 10_smooth_gradients.py
  09_Final_improved_real-esrgan.py

Run-AimlDataBranch:
  aiml_06_artifacts_removal_lama.py
  aiml_07_generate_synthetic_backgrounds.py
  aiml_08_prepare_training_split.py

Run-Training:
  aiml_08_train_segmentation_demo.py        (CPU; GPU redirects to aiml_08_submit_gpu_training.sh)

Run-Inference:
  aiml_09_run_trained_model_transparent.py
```

---

## Technical Dependencies

| Component | Purpose |
| :--- | :--- |
| PowerShell (`Get-ChildItem`, `Read-Host`, `Write-Host`, built-in) | Menu interaction, script invocation, and colour-coded console reporting. |
| `py` or `python` launcher on `PATH` | Required to execute the pipeline's `.py` steps; the script auto-detects which is available. |
| Underlying pipeline scripts (`00`–`10`, `aiml_06`–`aiml_09`) | Each must be present in the same `Scripts` folder as `run_pipeline.ps1`; missing scripts are skipped with a warning rather than causing a crash. |

---

## Execution Examples

### Launch the Interactive Menu
```powershell
.\run_pipeline.ps1
```

### One-Time Execution Policy Setup (if required)
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Typical Session
```text
================================================
 Project 2 Pipeline - Run Menu
================================================
 1) Legacy core            (00-05: video -> cropped frames)
 2) Legacy visual pipeline (06-10: classical + optional AI upscale)
 3) AIML data branch       (aiml_06 - aiml_08 split)
 4) Train model            (aiml_08 train)
 5) Run inference          (aiml_09)
 6) Run EVERYTHING in order (1 -> 2 and/or 3 -> 4 -> 5)
 0) Exit
================================================
Choose an option: 6
```

---

## Notes on Future Development

The script's closing comments document a planned later deployment phase: once the pipeline is stable, a friendlier front-end (e.g. a Flask/Streamlit web UI or a compiled desktop app) could wrap the same `Run-LegacyCore`, `Run-LegacyVisual`, `Run-AimlDataBranch`, `Run-Training`, and `Run-Inference` functions for users less comfortable with the command line. These functions are deliberately kept separate and single-purpose to support that reuse without rewriting the underlying orchestration logic.
