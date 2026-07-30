# Script Documentation: 05_extract_crop_frames_from_avi.ps1

## Overview

| Field | Value |
|---|---|
| **Script Name** | 05_extract_crop_frames_from_avi.ps1 |
| **Language** | PowerShell |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Frame Extraction & Cropping / Data Preparation / Automation |
| **Platform** | Windows (PowerShell), external dependency on `ffmpeg.exe` (FFmpeg suite) |
| **Execution Type** | Standalone, run manually or scheduled; supports parameters and a `-NoLog` switch |

---

## Categories

- **Video Decomposition & Region-of-Interest Cropping** — Extracts every frame of a source video and simultaneously crops each frame to a fixed pixel region using `ffmpeg`.
- **Data Preparation for AI/ML** — Produces a spatially-cropped, per-frame image dataset targeted at the specific region of the experimental capture that matters for downstream analysis.
- **Automation / Scripting** — Fully automated, parameterized (`-VideoFileName`, `-CropFilter`, `-FramePattern`, `-NoLog`), no interactive input required.
- **Logging & Reporting** — Writes a timestamped, leveled log file capturing extraction counts and outcomes, with careful handling of FFmpeg's stderr diagnostic output.

---

## Script Purpose

The script extracts every frame of a source video (default: `active.avi`) and crops each extracted frame to a fixed region using `ffmpeg.exe`, saving the result as individual PNG images. It:

1. Resolves its own script directory (`$PSScriptRoot`, with a fallback for older PowerShell versions) and derives the project root one level up.
2. Accepts a configurable video file name (`-VideoFileName`), a crop filter definition (`-CropFilter`, default `crop=940:600:0:40` — width 940, height 600, starting at x=0, y=40), an output frame filename pattern (`-FramePattern`, default `frame_%04d.png`), and an optional `-NoLog` switch to disable file-based logging.
3. Builds paths to the source video (`Video\`), `ffmpeg.exe` (`Bin\`), the cropped-frame output folder (`Images\02_Frames`), and the `Logs` folder.
4. Creates the `Images\02_Frames` output folder (always) and the `Logs` folder (only if logging is enabled) if they don't already exist.
5. Verifies that both `ffmpeg.exe` and the source video file exist before proceeding, logging file size and last-modified time for the video.
6. Counts existing `frame_*.png` files in the output folder **before** extraction, to later gauge how many new frames were produced.
7. Runs `ffmpeg` with the `-vf` (video filter) option set to the crop filter, extracting and cropping every frame in a single pass, and writing PNG files according to the naming pattern.
8. **Correctly handles FFmpeg's normal stderr output**: FFmpeg writes version and progress information to stderr on every run, even when successful. The script temporarily sets `$ErrorActionPreference = "Continue"` around the `ffmpeg` call (restoring it immediately afterward) so this routine stderr output is captured to a diagnostic file rather than being misinterpreted by PowerShell as a terminating error — and instead relies on `$LASTEXITCODE` to determine actual success or failure.
9. Logs the FFmpeg exit code explicitly, and only raises an error (including the captured diagnostic output) if the exit code is non-zero.
10. Counts frame files **after** extraction, logs the before/after counts and the number of new frames created, and notes (without treating it as an error) if diagnostic output was captured.
11. Logs every step (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) to `Logs\05_extract_crop_frames_from_avi.log`, unless logging is disabled, and always echoes log entries to the console.
12. Cleans up its temporary FFmpeg diagnostic file on success, and exits with code `1` on any genuine failure.

---

## Business / Process Purpose

This script performs a targeted, region-specific data-preparation step for **Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**, producing a cropped frame dataset focused on the specific portion of the video frame relevant to the experiment. Its business drivers include:

- **Focused AI/ML training input**: Rather than passing full, uncropped frames into the super-resolution/reconstruction pipeline, this script isolates a fixed pixel region (`940×600`, offset from the top-left by `0,40`) — presumably the region of the mixing tank or imaging field of interest — reducing irrelevant background/border content and standardizing input dimensions for model training and inference.
- **Consistent, reproducible cropping**: Because the crop geometry is defined as a script parameter with a fixed default, every run applies an identical crop, ensuring frame-to-frame and run-to-run consistency across the experimental dataset — important for model training stability and for fair comparison between reconstructed and ground-truth data.
- **Reliable automation despite noisy tool output**: FFmpeg's normal behavior of writing diagnostic information to stderr had previously caused this type of script to fail unnecessarily under PowerShell's strict error handling. Correctly separating "FFmpeg wrote to stderr" from "FFmpeg actually failed" (via the `$LASTEXITCODE` check) ensures the automated pipeline runs reliably in production/research use rather than generating false failures that require manual intervention.
- **Auditable dataset generation**: Logged before/after frame counts and exit-code confirmation give the research team a verifiable record of how the cropped dataset was produced from a given source video, supporting reproducibility of experimental results.

---

## Logging the Result

Because this script produces the spatially-cropped image dataset that will directly feed the AI/ML super-resolution and volumetric reconstruction pipeline, its log is an important record of **how and from what region the training/inference dataset was generated**.

- **Log file location**: `<ScriptDir>\Logs\05_extract_crop_frames_from_avi.log` (created only when logging is enabled; disabled via `-NoLog`)
- **Log format**: `[yyyy-MM-dd HH:mm:ss] [LEVEL] Message`
- **Log levels used**: `INFO`, `SUCCESS`, `WARNING`, `ERROR`
- **Log content captured**:
  - Script start/completion markers and script/project identification
  - Resolved script directory, project root, input video path, `ffmpeg` path, **the crop filter string used**, and output folder path
  - A `WARNING`-level note when file logging has been disabled via `-NoLog`
  - Input video file size and last-modified timestamp
  - The exact output frame filename pattern used
  - The number of frame files already present in the output folder **before** extraction
  - Confirmation that extraction started, and the explicit FFmpeg exit code returned
  - The frame count **before** and **after** extraction, and the number of new frames created during the run
  - The output folder path where cropped frames were written
  - A note confirming that any captured FFmpeg diagnostic output was normal (non-error) output, since the run succeeded
  - Full diagnostic detail from FFmpeg's stderr stream if the process genuinely fails, including its exit code
- **Output artifact**: `<ProjectRoot>\Images\02_Frames\` — a folder of sequentially numbered, cropped PNG frame images (e.g., `frame_0001.png`, `frame_0002.png`, …), each reduced to the `940×600` region of interest, ready for consumption by downstream super-resolution or volumetric reconstruction steps.
- **Console output**: Every log entry is echoed to the console in real time via `Write-Host`, regardless of whether file logging is enabled — useful for interactive runs or scheduled-task output capture.
- **Research/ML pipeline value**: The explicit exit-code log entry and the before/after frame-count reporting together give a two-part confidence check — that FFmpeg genuinely succeeded (not just "didn't throw"), and that the expected number of frames was actually produced — both of which matter when the resulting cropped dataset will be used to train or evaluate a model.

---

## Technology Assessment

| Aspect | Assessment |
|---|---|
| **Complexity** | Moderate. Parameterized crop geometry, a reusable logging function, before/after frame counting, explicit exit-code handling, and a deliberate workaround for PowerShell's stderr-as-error behavior. |
| **Dependencies** | Requires `ffmpeg.exe` (part of the FFmpeg toolset) to be present at `Bin\ffmpeg.exe` relative to the project root. No PowerShell modules beyond built-ins are required. |
| **Error Handling** | Strong, and notably more refined than a naive implementation — correctly distinguishes FFmpeg's routine stderr diagnostic output from genuine failures by relying on `$LASTEXITCODE` rather than the mere presence of stderr content, avoiding false-positive failures while still capturing full diagnostic detail when a real failure occurs. |
| **Portability** | Windows-only (PowerShell-native); depends on a specific relative folder layout (`Bin`, `Video`, `Images\02_Frames`, `Logs`) and a bundled/pre-installed `ffmpeg.exe`. Includes a fallback for `$PSScriptRoot` on older PowerShell versions. |
| **Idempotency** | Partial — FFmpeg is invoked with `-y` (overwrite), so re-running with the same frame pattern overwrites corresponding frame numbers. However, since the output folder is not cleared before extraction, if a prior run produced more frames than the current source video, **stale leftover frames beyond the new count could remain** in the output folder. |
| **Security Considerations** | Executes a local external binary (`ffmpeg.exe`) with mostly fixed arguments (crop geometry is parameterized but not derived from untrusted external input in normal use) against a fixed input path — low risk of command injection. No network calls, no credential handling. As with any script invoking an external executable, the provenance and integrity of `ffmpeg.exe` should be verified (e.g., checksummed/sourced from an official FFmpeg release) before deployment. |
| **Performance** | Resource-intensive relative to metadata-only operations — full frame decoding, cropping, and PNG encoding of an entire video can be CPU- and disk-I/O-heavy, generating a large number of output files (potentially thousands for longer videos) with corresponding disk space requirements. |
| **Maintainability** | High — parameterized design (`-VideoFileName`, `-CropFilter`, `-FramePattern`, `-NoLog`), a reusable `Write-Log` function, and clear inline documentation and comments (including an explanation of the stderr-handling logic) make the script easy to understand, extend, and troubleshoot. |

---

## Approval Considerations

- **Least privilege**: Requires read access to the video file and `ffmpeg.exe`, and write access to the `Images\02_Frames` output folder and (optionally) `Logs` — no elevated/administrative rights needed.
- **Third-party binary dependency**: The script depends on an external executable (`ffmpeg.exe`). Approvers should confirm the binary's source, version, and licensing (FFmpeg is LGPL/GPL depending on build configuration) are acceptable for the environment, and that it is deployed from a trusted, verified source.
- **Crop geometry validation**: The crop filter (`crop=940:600:0:40`) defines a fixed pixel region of interest. Approvers should confirm this region is correct for the current camera/capture setup — an incorrect crop would silently produce a dataset missing the intended experimental content, without the script itself detecting or flagging that condition (it only validates that FFmpeg ran successfully, not that the crop is meaningful).
- **Storage impact**: Frame extraction can generate a very large number of PNG files and consume significant disk space, especially for longer or higher-resolution/higher-frame-rate source videos. Approvers should confirm adequate storage capacity and, if needed, a retention/cleanup policy for the `Images\02_Frames` folder.
- **Stale output risk**: Because the output folder is not cleared before extraction, re-running the script against a shorter video (or a different video under the same pattern) could leave stale frames from a previous run mixed in with new ones. Approvers should consider whether an explicit folder-clearing step or a run-specific/timestamped subfolder is warranted.
- **No external communication**: `ffmpeg` is run entirely locally against a local file; no network calls or data leave the machine.
- **No destructive actions on the source**: The source video is only read, never modified, moved, or deleted.
- **Auditability**: Structured, leveled logging (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) with explicit exit-code confirmation and before/after frame counts supports change-control, QA, and audit requirements out of the box.
- **Path and environment assumptions**: Approvers should confirm the expected folder structure (`ProjectRoot\Bin\ffmpeg.exe`, `ProjectRoot\Video\<file>`, `ProjectRoot\Images\02_Frames`, `ScriptDir\Logs`) matches the target deployment environment.
- **Logging opt-out**: The `-NoLog` switch allows disabling persistent logging entirely (the script does log a `WARNING` about this to the console). Approvers should determine whether disabling logging is acceptable for production/research runs, given the provenance value of the log for this project.

---

## Suggested Classification

**Classification: Low-to-Moderate Risk / Standard Automation Utility with a Managed External Dependency and Storage Impact**

- **Risk Level**: Low-to-moderate — read-only interaction with source data, no network activity, no credential handling; risk factors beyond a simple metadata script include reliance on a third-party executable (`ffmpeg.exe`), the potential for significant disk space consumption, stale-output accumulation if run repeatedly without folder cleanup, and the possibility of an incorrect crop region silently producing an unintended dataset.
- **Change Category**: Standard/pre-approved change candidate for routine or scheduled execution, contingent on the `ffmpeg.exe` binary being separately vetted/approved as part of the toolchain, the crop geometry being validated for the current capture setup, and storage capacity/retention being addressed operationally.
- **Data Sensitivity**: `active.avi` is understood (from the broader project context) to contain scientific imaging data — captures related to volumetric/mixing-tank experiments intended for AI/ML super-resolution and reconstruction work — rather than generic video content. This script's output (cropped PNG frames) is a direct, spatially-focused derivative of that same experimental content:
  - The `Images\02_Frames` cropped frame set effectively **is** the experimental image data, narrowed to the region of interest. Any access controls, storage protections, or confidentiality requirements that apply to the source video should be treated as applying equally to this cropped frame output.
  - If the underlying video is subject to research data management, IP/pre-publication confidentiality, or (where applicable) human/animal subject research policies, the **process** around this script — storage of `active.avi` and the cropped frames, access to `Bin`, `Video`, `Images`, and `Logs` folders, and retention of frame/log history — should be classified and governed according to the project's data management plan, even though the script itself remains low-to-moderate risk from a purely technical standpoint.
- **Recommended Handling**: Approve as a routine pipeline utility, contingent on verifying the `ffmpeg.exe` dependency, confirming the crop geometry is correct for the current experimental setup, and addressing storage/retention and stale-frame considerations operationally; classify and protect the cropped frame images with the same sensitivity level established for the Project 2 experimental video dataset.
