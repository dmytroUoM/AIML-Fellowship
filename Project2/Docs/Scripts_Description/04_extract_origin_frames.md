# Script Description: 04_extract_origin_frames.ps1

## Overview

| Field | Value |
|---|---|
| **Script Name** | 04_extract_origin_frames.ps1 |
| **Language** | PowerShell |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Frame Extraction / Data Preparation / Automation |
| **Platform** | Windows (PowerShell), external dependency on `ffmpeg.exe` (FFmpeg suite) |
| **Execution Type** | Standalone, run manually or scheduled; supports parameters and a `-NoLog` switch |

---

## Categories

- **Video Decomposition / Frame Extraction** — Converts a source video into a sequence of individual still-image frames using `ffmpeg`.
- **Data Preparation for AI/ML** — Produces the raw per-frame image dataset that will be consumed by downstream super-resolution and volumetric reconstruction models.
- **Automation / Scripting** — Fully automated, parameterized (`-VideoFileName`, `-FramePattern`, `-NoLog`), no interactive input required.
- **Logging & Reporting** — Writes a timestamped, leveled log file capturing extraction counts and outcomes, with error handling built in.

---

## Script Purpose

The script extracts every frame of a source video (default: `active.avi`) as individual PNG images using `ffmpeg.exe`. It:

1. Resolves its own script directory (`$PSScriptRoot`, with a fallback for older PowerShell versions) and derives the project root one level up.
2. Accepts a configurable video file name (`-VideoFileName`), an output frame filename pattern (`-FramePattern`, default `frame_%04d.png`), and an optional `-NoLog` switch to disable file-based logging.
3. Builds paths to the source video (`Video\`), `ffmpeg.exe` (`Bin\`), the frame output folder (`Images\01_Origin`), and the `Logs` folder.
4. Creates the `Images\01_Origin` output folder (always) and the `Logs` folder (only if logging is enabled) if they don't already exist.
5. Verifies that both `ffmpeg.exe` and the source video file exist before proceeding, logging file size and last-modified time for the video.
6. Counts existing PNG files in the output folder **before** extraction, to later gauge how many new frames were produced.
7. Runs `ffmpeg` (with `-y` to overwrite existing output files) to decode the video and write out one PNG per frame according to the naming pattern, redirecting `stderr` to a temporary error file for diagnostics.
8. Checks the `ffmpeg` exit code and raises an error with captured `stderr` content if extraction fails.
9. Counts PNG files **after** extraction and logs the approximate number of new frames created during the run.
10. Logs every step (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) to `Logs\04_extract_origin_frames.log`, unless logging is disabled, and always echoes log entries to the console.
11. Cleans up its temporary `ffmpeg` stderr file on success, and exits with code `1` on any failure.

---

## Business / Process Purpose

This script performs a foundational data-preparation step for **Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**, converting a continuous experimental video capture into a discrete, per-frame image dataset that AI/ML models can consume. Its business drivers include:

- **AI/ML training data generation**: Super-resolution and volumetric reconstruction models operate on individual images or image sequences, not raw video files. This script produces the "origin" (source-resolution, unmodified) frame set that anchors the rest of the processing pipeline - later steps use these frames for feeding a reconstruction algorithm.
- **Traceable, repeatable dataset creation**: By logging exact frame counts (before/after) and the output pattern used, the script creates a verifiable record of how many frames were generated from a given source video and where they were placed, supporting reproducibility of the experimental dataset.
- **Consistent naming and storage convention**: Writing frames into a dedicated `Images\01_Origin` folder using a predictable zero-padded naming pattern (`frame_%04d.png`) ensures frames can be reliably ordered, referenced, and matched against downstream outputs (e.g., a corresponding `02_SuperResolved` or reconstruction output folder).
- **Early failure detection**: Validating `ffmpeg`'s exit code and capturing its error output prevents a failed or partial extraction from silently producing an incomplete frame set that could compromise model training or reconstruction accuracy.

---

## Logging the Result

Because this script produces the foundational image dataset for the AI/ML super-resolution and volumetric reconstruction pipeline, its log is a key record of **how the experimental dataset was generated** for a given source video.

- **Log file location**: `<ScriptDir>\Logs\04_extract_origin_frames.log` (created only when logging is enabled; disabled via `-NoLog`)
- **Log format**: `[yyyy-MM-dd HH:mm:ss] [LEVEL] Message`
- **Log levels used**: `INFO`, `SUCCESS`, `WARNING`, `ERROR`
- **Log content captured**:
  - Script start/completion markers and script/project identification
  - Resolved script directory, project root, input video path, `ffmpeg` path, output folder path
  - A `WARNING`-level note when file logging has been disabled via `-NoLog`
  - Input video file size and last-modified timestamp
  - The exact output frame filename pattern used
  - The number of PNG frames already present in the output folder **before** extraction (relevant if frames from a prior run were not cleared out)
  - Confirmation that `ffmpeg` extraction started
  - The total PNG frame count in the output folder **after** extraction, and the approximate number of new frames created during the run
  - The path where frames were written
  - Full error detail captured from `ffmpeg`'s stderr stream if the process fails, including its exit code
- **Output artifact**: `<ProjectRoot>\Images\01_Origin\` — a folder of sequentially numbered PNG frame images (e.g., `frame_0001.png`, `frame_0002.png`, …) representing the complete decomposition of the source video, ready for consumption by downstream super-resolution or volumetric reconstruction steps.
- **Console output**: Every log entry is echoed to the console in real time via `Write-Host`, regardless of whether file logging is enabled — useful for interactive runs or scheduled-task output capture.
- **Research/ML pipeline value**: Because "before/after" frame counts are logged rather than just a simple success message, the log itself provides a lightweight audit trail for detecting incomplete extractions, accidental re-runs into a non-empty folder, or unexpected frame-count discrepancies — all of which matter when the resulting frames will be used to train or evaluate a model.

---

## Technology Assessment

| Aspect | Assessment |
|---|---|
| **Complexity** | Moderate. Parameterized inputs, a reusable logging function, before/after frame counting, and structured error capture from an external decoding process. |
| **Dependencies** | Requires `ffmpeg.exe` (part of the FFmpeg toolset) to be present at `Bin\ffmpeg.exe` relative to the project root. No PowerShell modules beyond built-ins are required. |
| **Error Handling** | Solid — `$ErrorActionPreference = "Stop"`, wrapped in `try/catch`, explicit checks for `ffmpeg.exe` and video file existence, exit-code checking, and captured stderr detail surfaced in the thrown error. However, unlike a pure validation step, there is no check that the *number* of extracted frames is reasonable or non-zero beyond what `ffmpeg`'s own exit code reports. |
| **Portability** | Windows-only (PowerShell-native); depends on a specific relative folder layout (`Bin`, `Video`, `Images\01_Origin`, `Logs`) and a bundled/pre-installed `ffmpeg.exe`. Includes a fallback for `$PSScriptRoot` on older PowerShell versions. |
| **Idempotency** | Partial — `ffmpeg` is invoked with `-y` (overwrite), so re-running with the same frame pattern will overwrite corresponding frame numbers. However, if the source video length changes or a prior run left extra frames beyond the new frame count, **stale leftover frames could persist** in the output folder, since the script does not clear the folder before extraction. This is a notable operational consideration rather than a strict idempotency guarantee. |
| **Security Considerations** | Executes a local external binary (`ffmpeg.exe`) with mostly fixed arguments against a fixed input path — low risk of command injection. No network calls, no credential handling. As with any script invoking an external executable, the provenance and integrity of `ffmpeg.exe` should be verified (e.g., checksummed/sourced from an official FFmpeg release) before deployment. |
| **Performance** | Resource-intensive relative to metadata-only operations — full frame decoding and PNG encoding of an entire video can be CPU- and disk-I/O-heavy and time-consuming, and will generate a large number of output files (potentially thousands for longer videos), with corresponding disk space requirements. |
| **Maintainability** | High — parameterized design (`-VideoFileName`, `-FramePattern`, `-NoLog`), a reusable `Write-Log` function, and clear inline documentation (including a purpose block in the header) make the script easy to understand and extend. |

---

## Approval Considerations

- **Least privilege**: Requires read access to the video file and `ffmpeg.exe`, and write access to the `Images\01_Origin` output folder and (optionally) `Logs` — no elevated/administrative rights needed.
- **Third-party binary dependency**: The script depends on an external executable (`ffmpeg.exe`). Approvers should confirm the binary's source, version, and licensing (FFmpeg is LGPL/GPL depending on build configuration) are acceptable for the environment, and that it is deployed from a trusted, verified source.
- **Storage impact**: Frame extraction can generate a very large number of PNG files and consume significant disk space, especially for longer or higher-resolution/higher-frame-rate source videos. Approvers should confirm adequate storage capacity and, if needed, a retention/cleanup policy for the `Images\01_Origin` folder.
- **Stale output risk**: Because the output folder is not cleared before extraction and `ffmpeg` only overwrites files with matching output names, re-running the script against a shorter video (or a different video under the same pattern) could leave stale frames from a previous run mixed in with new ones. Approvers should consider whether the script needs an explicit folder-clearing step or a run-specific/timestamped subfolder to avoid ambiguous datasets.
- **No external communication**: `ffmpeg` is run entirely locally against a local file; no network calls or data leave the machine.
- **No destructive actions on the source**: The source video is only read, never modified, moved, or deleted.
- **Auditability**: Structured, leveled logging (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) with before/after frame counts supports change-control, QA, and audit requirements out of the box.
- **Path and environment assumptions**: Approvers should confirm the expected folder structure (`ProjectRoot\Bin\ffmpeg.exe`, `ProjectRoot\Video\<file>`, `ProjectRoot\Images\01_Origin`, `ScriptDir\Logs`) matches the target deployment environment.
- **Logging opt-out**: The `-NoLog` switch allows disabling persistent logging entirely (the script does log a `WARNING` about this to the console). Approvers should determine whether disabling logging is acceptable for production/research runs, given the provenance value of the log for this project.

---

## Suggested Classification

**Classification: Low-to-Moderate Risk / Standard Automation Utility with a Managed External Dependency and Storage Impact**

- **Risk Level**: Low-to-moderate — read-only interaction with source data, no network activity, no credential handling; risk factors beyond a simple metadata script include reliance on a third-party executable (`ffmpeg.exe`) and the potential for significant disk space consumption and stale-output accumulation if run repeatedly without folder cleanup.
- **Change Category**: Standard/pre-approved change candidate for routine or scheduled execution, contingent on the `ffmpeg.exe` binary being separately vetted/approved as part of the toolchain, and on storage capacity/retention being addressed operationally.
- **Data Sensitivity**: `active.avi` is understood (from the broader project context) to contain scientific imaging data — captures related to volumetric/mixing-tank experiments intended for AI/ML super-resolution and reconstruction work — rather than generic video content. This script's output (individual PNG frames) is a direct decomposition of that same experimental content, so the same sensitivity considerations apply directly to the extracted frames, not just the source video:
  - The `Images\01_Origin` frame set effectively **is** the experimental image data in a different, more directly usable form (individual images rather than an encoded video container). Any access controls, storage protections, or confidentiality requirements that apply to the source video should be treated as applying equally to this frame output.
  - If the underlying video is subject to research data management, IP/pre-publication confidentiality, or (where applicable) human/animal subject research policies, the **process** around this script — storage of `active.avi` and the extracted frames, access to `Bin`, `Video`, `Images`, and `Logs` folders, and retention of frame/log history — should be classified and governed according to the project's data management plan, even though the script itself remains low-to-moderate risk from a purely technical standpoint.
- **Recommended Handling**: Approve as a routine pipeline utility, contingent on verifying the `ffmpeg.exe` dependency and addressing storage/retention and stale-frame considerations operationally; classify and protect the extracted frame images with the same sensitivity level established for the Project 2 experimental video dataset.
