# Script Description: 01_Get_Avi_SHA256_Hash.ps1

## Overview

| Field | Value |
|---|---|
| **Script Name** | 01_Get_Avi_SHA256_Hash.ps1 |
| **Language** | PowerShell |
| **Category** | File Integrity / Data Verification / Automation |
| **Platform** | Windows (PowerShell) |
| **Execution Type** | Standalone, run manually or scheduled (e.g., via Task Scheduler) |

---

## Categories

- **Integrity Verification** — Generates a cryptographic hash used to confirm a file has not been altered or corrupted.
- **Digital Forensics / Chain of Custody Support** — Produces a verifiable fingerprint of a video file, commonly needed for evidentiary or audit trail purposes.
- **Automation / Scripting** — Fully automated, no user interaction required once triggered.
- **Logging & Reporting** — Writes both a persistent log file and a discrete report/output file.

---

## Script Purpose

The script calculates a **SHA256 cryptographic hash** of a video file named `active.avi`, located in a `Video` subfolder relative to the script's project root. It:

1. Resolves its own location (`$PSScriptRoot`) and derives the project root one level up.
2. Confirms (or creates) `Reports` and `Logs` directories.
3. Verifies that the target video file exists before proceeding.
4. Computes the SHA256 hash of the file using PowerShell's built-in `Get-FileHash` cmdlet.
5. Writes the hash value only (no extra metadata) to `Reports\01_avi_hash.txt`.
6. Logs every step of execution — including start/end times, resolved paths, and the computed hash — to `Logs\01_avi_hash.log`.
7. Exits with an error code (`exit 1`) and logs an `ERROR`-level entry if any step fails (e.g., missing file).

---

## Business / Process Purpose

This script supports a process where a video file (`active.avi`) needs a **verifiable, tamper-evident fingerprint** recorded at a point in time. Typical business drivers include:

- **Evidentiary integrity**: Establishing proof that a specific video file existed in a specific state at a specific time (e.g., surveillance footage, incident recordings, compliance recordings).
- **Change/tamper detection**: A stored hash can later be recomputed and compared to detect any modification, corruption, or substitution of the file.
- **Audit trail creation**: The combination of a timestamped log and a discrete hash report creates a simple, repeatable audit record suitable for internal review or external audit.
- **Pipeline/step tracking**: The "01_" prefix in the filename and outputs suggests this is the first step in a larger, numbered multi-script workflow (likely followed by scripts such as archiving, transcoding, upload, or notification steps).

---

## Logging the Result

- **Log file location**: `<ScriptDir>\Logs\01_avi_hash.log`
- **Log format**: `[yyyy-MM-dd HH:mm:ss] [LEVEL] Message`
- **Log levels used**: `INFO` (normal operation) and `ERROR` (failure)
- **Log content captured**:
  - Script start/completion markers
  - Resolved script directory, project root, video file path, reports/logs folder paths
  - Confirmation that hashing began
  - The computed hash value
  - Path where the hash was saved
  - Any exception message on failure
- **Output artifact**: `<ProjectRoot>\Reports\01_avi_hash.txt` — contains **only the raw hash string**, making it easy to consume programmatically by downstream steps or comparison scripts.
- **Console output**: Every log entry is also written to the console (`Write-Host`) in real time, useful for interactive runs or monitoring via a scheduled task's output capture.

---

## Technology Assessment

| Aspect | Assessment |
|---|---|
| **Complexity** | Low. Linear script, single responsibility, no external dependencies. |
| **Dependencies** | None beyond built-in PowerShell cmdlets (`Get-FileHash`, `Test-Path`, `New-Item`, `Add-Content`, `Set-Content`). No third-party modules required. |
| **Error Handling** | Present and reasonable — uses `$ErrorActionPreference = "Stop"`, a `try/catch` block, explicit file-existence check, and a non-zero exit code on failure. |
| **Portability** | Windows-only (PowerShell-native). Relies on relative folder structure (`$PSScriptRoot`), so it is portable across machines as long as the folder layout is preserved. |
| **Idempotency** | Yes — re-running simply overwrites the hash file and appends a new log entry; no destructive side effects. |
| **Security Considerations** | Uses SHA256, a currently secure and collision-resistant hashing algorithm suitable for integrity verification. Script does not handle credentials, does not make network calls, and does not execute remote or dynamic code. Low attack surface. |
| **Performance** | Efficient for typical video file sizes; `Get-FileHash` streams the file rather than loading it fully into memory. Very large files (many GB) may take noticeable time but will not fail on memory grounds. |
| **Maintainability** | High — code is short, well-commented, and uses clear variable names. Numbered naming convention (`01_`) suggests it's part of a documented, ordered pipeline. |

---

## Approval Considerations

- **Least privilege**: Script only requires read access to the video file and write access to `Reports` and `Logs` folders — no elevated/administrative rights needed.
- **No external communication**: Script performs no network calls, API calls, or data exfiltration; all activity is local to the file system.
- **No destructive actions**: Script does not delete, move, or modify the source video file — it only reads it to compute a hash.
- **Auditability**: Built-in logging supports change-control and audit requirements out of the box.
- **Path assumptions**: Approvers should confirm the expected folder structure (`ProjectRoot\Video\active.avi`, `ProjectRoot\Reports`, `ScriptDir\Logs`) matches the target deployment environment, since the script assumes a fixed relative layout.
- **Error handling on missing input**: Approvers should decide whether an `exit 1` on a missing video file is sufficient, or whether integration with an alerting/monitoring system (e.g., email, ticketing) is required for a production workflow.
- **File overwrite behavior**: Re-running the script overwrites the previous hash report without versioning or archiving the prior value — approvers may want to assess whether historical hash values need to be retained for compliance purposes.

---

## Suggested Classification

**Classification: Low Risk / Standard Automation Utility**

- **Risk Level**: Low — read-only interaction with source data, no network activity, no credential handling, no privileged operations.
- **Change Category**: Standard/pre-approved change candidate — suitable for routine or scheduled execution without case-by-case approval, once initial review is complete.
- **Data Sensitivity**: `active.avi` contains **scientific electrical impedance imaging (EIT) data captured during a research experiment**, rather than generic video content. This shifts the sensitivity profile from a generic "unknown video" concern to considerations specific to research data integrity and provenance:
  - **Research integrity / data provenance**: The hash acts as a scientific record — evidence that the raw impedance imaging data has not been altered since acquisition. This is directly relevant to reproducibility, peer review, and defending results against allegations of data manipulation, so the hash and its log should be treated as part of the experiment's official data record, not as a disposable utility output.
  - **Intellectual property / pre-publication confidentiality**: If the experiment is unpublished or tied to a grant, patent application, or proprietary method, the underlying imaging data may need controlled access even though the hash file itself (a short alphanumeric string) reveals nothing about the experiment's content or results.
  - **Attribution to human or biological subjects**: Electrical impedance imaging is frequently used in biomedical/physiological research (e.g., tissue, organ, or body-segment imaging). If the experiment involved human or animal subjects, the source video may fall under research ethics, IRB/IACUC, or data protection requirements (e.g., HIPAA, GDPR, or institutional research data policies), independent of the hashing script.
  - **Chain of custody for scientific evidence**: Because the hash is generated automatically and logged with a timestamp, this script can serve as part of a lightweight chain-of-custody mechanism supporting later claims about when the raw data was fixed and unaltered — valuable for lab notebooks, data audits, or regulatory submissions.
  - **Recommended handling**: The *process* around this script (secure storage of `active.avi`, restricted access to the `Video`, `Reports`, and `Logs` folders, and retention policy for hash/log history) should be classified and governed according to the experiment's data management plan and institutional research data policy — even though the script itself remains low-risk, since it only reads the file and writes a hash and a log entry.
- **Recommended Handling**: Approve as a routine utility script; treat any associated video content and its Reports/Logs artifacts according to the organization's data classification policy for the underlying video content.
