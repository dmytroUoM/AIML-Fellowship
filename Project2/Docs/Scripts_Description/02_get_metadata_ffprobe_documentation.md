# Script Documentation: 02_get_metadata_ffprobe.ps1

## Overview

| Field | Value |
|---|---|
| **Script Name** | 02_get_metadata_ffprobe.ps1 |
| **Language** | PowerShell |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Metadata Extraction / Data Characterization / Automation |
| **Platform** | Windows (PowerShell), external dependency on `ffprobe.exe` (FFmpeg suite) |
| **Execution Type** | Standalone, run manually or scheduled; supports parameters and a `-NoLog` switch |

---

## Categories

- **Technical Metadata Extraction** — Captures container/codec/stream-level details of a video file using `ffprobe`.
- **Data Characterization / Quality Control** — Documents the exact technical properties of the experimental capture (resolution, frame rate, codec, duration, etc.) used later in AI/ML processing.
- **Automation / Scripting** — Fully automated, parameterized (`-VideoFileName`, `-NoLog`), no interactive input required.
- **Logging & Reporting** — Writes a structured JSON report and an optional, timestamped log file, with validation and error-handling built in.

---

## Script Purpose

The script extracts **technical metadata** from a video file (default: `active.avi`) using `ffprobe.exe`. It:

1. Resolves its own script directory (`$PSScriptRoot`, with a fallback for older PowerShell versions) and derives the project root one level up.
2. Accepts a configurable video file name (`-VideoFileName`) and an optional `-NoLog` switch to disable file-based logging.
3. Builds paths to `ffprobe.exe` (in `Bin\`), the source video (in `Video\`), the `Reports` output folder, and the `Logs` output folder.
4. Creates the `Reports` folder (always) and the `Logs` folder (only if logging is enabled) if they don't already exist.
5. Verifies that both `ffprobe.exe` and the target video file exist before proceeding, logging file size and last-modified time for the video.
6. Runs `ffprobe` with `-show_format` and `-show_streams` in JSON output mode, redirecting `stderr` to a temporary error file for diagnostics.
7. Checks the `ffprobe` exit code, confirms the output is non-empty, and **validates the output as parseable JSON** before saving it.
8. Saves the validated JSON metadata to `Reports\02_avi_metadata_ffprobe.json`.
9. Logs every step (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) to `Logs\02_get_metadata_ffprobe.log`, unless logging is disabled, and always echoes log entries to the console.
10. Cleans up its temporary `ffprobe` stderr file on success, and exits with code `1` on any failure.

---

## Business / Process Purpose

This script supports **Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**, an effort to apply AI/ML techniques to reconstruct volumetric detail from experimental imaging data of mixing-tank behavior. Its business drivers include:

- **Model input characterization**: AI/ML super-resolution and volumetric reconstruction pipelines are sensitive to exact input video characteristics (frame rate, resolution, pixel format, codec, duration). Capturing this metadata automatically ensures every training/inference input is documented and comparable.
- **Reproducibility and dataset versioning**: The JSON metadata record lets researchers later confirm exactly what a given video file contained technically, at the time it was captured — supporting reproducible experiments and defensible research results.
- **Early error detection**: By validating `ffprobe`'s exit code and JSON structure before saving, the script prevents malformed or partial metadata from silently entering the pipeline and corrupting downstream processing or model training.
- **Process traceability**: Consistent, structured logging and a dedicated Reports output give the research/engineering team a repeatable, auditable way to confirm that metadata extraction ran successfully for a given experimental video before it moves further into the AI/ML workflow.

---

## Logging the Result

Because this script feeds directly into the AI/ML data pipeline for the mixing-tank volumetric reconstruction project, its log and JSON report together form part of the **provenance and quality-control record** for each experimental video used in model training or inference.

- **Log file location**: `<ScriptDir>\Logs\02_get_metadata_ffprobe.log` (created only when logging is enabled; disabled via `-NoLog`)
- **Log format**: `[yyyy-MM-dd HH:mm:ss] [LEVEL] Message`
- **Log levels used**: `INFO`, `SUCCESS`, `WARNING`, `ERROR`
- **Log content captured**:
  - Script start/completion markers and script/project identification
  - Resolved script directory, project root, `ffprobe` path, video file path, Reports/Logs folder paths
  - Video file size and last-modified timestamp
  - Confirmation that `ffprobe` extraction started
  - JSON validation result (`SUCCESS` on valid JSON, `ERROR` with exception detail if invalid)
  - Path where the metadata file was saved
  - Full error detail captured from `ffprobe`'s stderr stream if the process fails, including its exit code
- **Output artifact**: `<ProjectRoot>\Reports\02_avi_metadata_ffprobe.json` — a complete `ffprobe` format/stream JSON dump (container format, duration, bit rate, codec, resolution, frame rate, audio/video stream details, etc.), consumable directly by downstream scripts or ML data-loading pipelines.
- **Console output**: Every log entry is echoed to the console in real time via `Write-Host`, regardless of whether file logging is enabled — useful for interactive runs or scheduled-task output capture.
- **Research/ML pipeline value**: Because model training and volumetric reconstruction outcomes are sensitive to exact capture parameters, retaining this metadata report and log (rather than treating them as transient build artifacts) supports experiment reproducibility, dataset versioning, and the ability to audit exactly what technical configuration produced a given trained model or reconstruction result.

---

## Technology Assessment

| Aspect | Assessment |
|---|---|
| **Complexity** | Low-to-moderate. Includes parameters, a reusable logging function, JSON validation, and structured error capture from an external process. |
| **Dependencies** | Requires `ffprobe.exe` (part of the FFmpeg toolset) to be present at `Bin\ffprobe.exe` relative to the project root. No PowerShell modules beyond built-ins are required. |
| **Error Handling** | Strong — `$ErrorActionPreference = "Stop"`, wrapped in `try/catch`, explicit checks for `ffprobe.exe` and video file existence, exit-code checking, empty-output checking, and JSON schema validation before persisting output. Cleans up temporary error files on success. |
| **Portability** | Windows-only (PowerShell-native); depends on a specific relative folder layout (`Bin`, `Video`, `Reports`, `Logs`) and a bundled/pre-installed `ffprobe.exe`. Includes a fallback for `$PSScriptRoot` on older PowerShell versions, improving compatibility. |
| **Idempotency** | Yes — re-running overwrites the existing JSON metadata report and appends a new log entry; no destructive side effects on the source video. |
| **Security Considerations** | Executes a local external binary (`ffprobe.exe`) with fixed, non-user-controlled flags against a fixed input path — low risk of command injection. No network calls, no credential handling. As with any script invoking an external executable, the provenance and integrity of `ffprobe.exe` itself should be verified (e.g., checksummed/sourced from an official FFmpeg release) before deployment. |
| **Performance** | Fast — `ffprobe` reads container/stream headers rather than decoding the full video, so metadata extraction is lightweight even for large experimental video files. |
| **Maintainability** | High — parameterized design (`-VideoFileName`, `-NoLog`), a reusable `Write-Log` function, and clear inline documentation (including an ASCII folder-structure diagram in the header) make the script easy to understand and extend. |

---

## Approval Considerations

- **Least privilege**: Requires read access to the video file and `ffprobe.exe`, and write access to `Reports` and (optionally) `Logs` — no elevated/administrative rights needed.
- **Third-party binary dependency**: The script depends on an external executable (`ffprobe.exe`). Approvers should confirm the binary's source, version, and licensing (FFmpeg is LGPL/GPL depending on build configuration) are acceptable for the environment, and that it is deployed from a trusted, verified source.
- **No external communication**: `ffprobe` is run entirely locally against a local file; no network calls or data leave the machine.
- **No destructive actions**: The source video is only read, never modified, moved, or deleted.
- **Auditability**: Structured, leveled logging (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) and JSON-schema-validated output support change-control, QA, and audit requirements out of the box.
- **Path and environment assumptions**: Approvers should confirm the expected folder structure (`ProjectRoot\Bin\ffprobe.exe`, `ProjectRoot\Video\<file>`, `ProjectRoot\Reports`, `ScriptDir\Logs`) matches the target deployment environment.
- **Logging opt-out**: The `-NoLog` switch allows disabling persistent logging entirely. Approvers should determine whether disabling logging is acceptable for production/research runs, given the provenance value of the log file for this project, or whether the switch should be restricted/disallowed in governed environments.
- **File overwrite behavior**: Re-running overwrites the previous metadata JSON without versioning — approvers may want to assess whether historical metadata snapshots need retention for reproducibility or compliance purposes.

---

## Suggested Classification

**Classification: Low Risk / Standard Automation Utility with a Managed External Dependency**

- **Risk Level**: Low — read-only interaction with source data, no network activity, no credential handling; the main added risk factor is reliance on a third-party executable (`ffprobe.exe`), which should be sourced and verified through a controlled process.
- **Change Category**: Standard/pre-approved change candidate for routine or scheduled execution, contingent on the `ffprobe.exe` binary itself being separately vetted/approved as part of the toolchain.
- **Data Sensitivity**: `active.avi` is understood (from the broader project context) to contain scientific imaging data — captures related to volumetric/mixing-tank experiments intended for AI/ML super-resolution and reconstruction work — rather than generic video content.
  - The extracted metadata (frame rate, resolution, codec, duration, etc.) is technical/non-content data and is low sensitivity in isolation, but it should still be retained and version-controlled as part of the experiment's data record, since AI/ML pipeline reproducibility depends on knowing the exact technical characteristics of each training/inference input.
  - If the underlying video is subject to research data management, IP/pre-publication confidentiality, or (where applicable) human/animal subject research policies, the **process** around this script — storage of `active.avi`, access to `Bin`, `Video`, `Reports`, and `Logs` folders, and retention of metadata/log history — should be classified and governed according to the project's data management plan, even though the script itself remains low-risk.
- **Recommended Handling**: Approve as a routine pipeline utility, contingent on verifying the `ffprobe.exe` dependency; classify and protect the surrounding video content, metadata reports, and logs according to the sensitivity level established for the Project 2 experimental dataset.
