# Script Description and Assessment: 03_get_metadata_powershell.ps1

## 1. Script Identification

| Item | Description |
|---|---|
| Script name | `03_get_metadata_powershell.ps1` |
| Project | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| Script type | Windows PowerShell automation script |
| Main input | `Video\active.avi` |
| Main output | `Reports\03_active-avi_metadata_powershell.txt` |
| Log output | `Scripts\Logs\03_get_metadata_powershell.log` |
| Primary technology | Windows PowerShell with `Shell.Application` COM object |
| Execution environment | Windows desktop or Windows PowerShell environment |

---

## 2. Script Categories

This script can be classified under the following categories:

- **Data preparation and validation**  
  The script extracts file-level and media-related metadata from the source AVI video before further processing.

- **AIML project evidence generation**  
  The metadata report supports reproducibility and traceability within the AIML-driven super-resolution and volumetric reconstruction workflow.

- **Digital asset inventory**  
  The script records key properties of the source video file, including file size, timestamps, and selected Windows Explorer metadata fields.

- **Process automation**  
  The script automates a repeatable manual task that would otherwise require checking video properties through Windows Explorer.

- **Audit logging and traceability**  
  The script creates a timestamped execution log showing the actions completed, paths used, success messages, warnings, and errors.

---

## 3. Script Purpose

The purpose of `03_get_metadata_powershell.ps1` is to extract selected metadata from the project video file `active.avi` using Windows PowerShell and the Windows `Shell.Application` COM object.

The script performs the following actions:

1. Identifies the script location using `$PSScriptRoot`.
2. Defines the project root as one folder above the `Scripts` directory.
3. Locates the input video file in the project `Video` folder.
4. Creates a `Reports` folder if it does not already exist.
5. Creates a `Logs` folder beside the script when logging is enabled.
6. Uses the Windows Shell metadata interface to inspect selected metadata indexes.
7. Saves the extracted metadata into a text report.
8. Writes a timestamped log of the execution process.
9. Handles errors using `try`, `catch`, and `finally` blocks.
10. Releases COM objects after execution to reduce resource locking or memory issues.

The script is especially useful where the project needs a simple, repeatable, Windows-native way to document metadata associated with the source AVI file.

---

## 4. Business / Process Purpose

The business and process purpose of this script is to support controlled and repeatable handling of source video data used in the AIML reconstruction workflow.

In the context of the mixing tank super-resolution and volumetric reconstruction project, the quality and identity of the source video are important because the video forms part of the input evidence for later processing stages. Recording metadata helps demonstrate that the correct input file was used and that the file properties were known at the point of processing.

The script supports the process by:

- documenting the source video before further AIML processing;
- creating a repeatable metadata extraction step;
- reducing manual checking through Windows Explorer;
- supporting project evidence and technical audit trails;
- helping identify changes to the source file, such as modification date or file size changes;
- providing supporting documentation for project reports, notebooks, or assessment evidence.

For professional engineering or technical documentation, this script contributes to traceability, repeatability, and data governance. It provides evidence that the source media file was checked and recorded before downstream analysis.

---

## 5. Logging the Result

The script includes a logging function named `Write-Log`. This function writes messages both to the PowerShell console and, when enabled, to a log file.

### Log file location

The log file is saved beside the script in the `Logs` folder:

```text
Scripts\Logs\03_get_metadata_powershell.log
```

### Report file location

The metadata report is saved in the project `Reports` folder:

```text
Reports\03_active-avi_metadata_powershell.txt
```

### Logging behaviour

By default, logging is enabled. The user can disable file logging by running the script with the `-NoLog` switch:

```powershell
.\03_get_metadata_powershell.ps1 -NoLog
```

When logging is enabled, the script records:

- script start;
- script name;
- project name;
- script folder path;
- project root path;
- video file path;
- reports folder path;
- logs folder path;
- resolved video path;
- video file size;
- video last modified date;
- metadata extraction status;
- number of metadata entries saved;
- final success or failure status.

The logging levels used by the script are:

| Level | Meaning |
|---|---|
| `INFO` | Normal process information |
| `WARNING` | Non-critical issue, such as no metadata returned or logging disabled |
| `ERROR` | Script failure or missing required file |
| `SUCCESS` | Successful completion of a key step |

This logging structure makes the script suitable for inclusion in a controlled project workflow because it records what happened, when it happened, and where the output was saved.

---

## 6. Technology Assessment

### Technology used

The script uses:

- Windows PowerShell;
- `$PSScriptRoot` for relative path handling;
- `Shell.Application` COM object for Windows Explorer metadata access;
- `Get-Item` and `Resolve-Path` for file validation;
- `Set-Content` and `Add-Content` for writing reports and logs;
- `try`, `catch`, and `finally` for error handling and cleanup.

### Strengths

- **No external dependency**  
  The script does not require `ffprobe.exe` or Python libraries. It uses Windows-native functionality.

- **Portable within the project structure**  
  Paths are built relative to the script location, which makes the script easier to move with the project folder.

- **Readable audit output**  
  The report is saved as a plain text file, which is easy to open, review, archive, or include in documentation.

- **Basic error control**  
  The script stops on errors and reports failures clearly through the logging function.

- **Resource cleanup**  
  COM objects are released in the `finally` block, reducing the risk of retained handles or memory issues.

### Limitations

- **Windows-only operation**  
  The script depends on `Shell.Application`, so it is not suitable for WSL Ubuntu, Linux, or macOS environments.

- **Metadata index dependency**  
  Windows Shell metadata indexes can vary depending on Windows version, file type, installed codecs, and Explorer configuration.

- **Limited technical media detail**  
  Compared with `ffprobe`, Windows Shell metadata may provide less detailed codec, stream, frame rate, and container information.

### Recommended use

This script is best used as a supporting metadata extraction step alongside the `ffprobe` metadata script. The PowerShell script provides Windows Explorer-style metadata, while `ffprobe` provides more complete technical media metadata.

---

## 7. Approval Considerations

Before approving this script for routine project use, the following points should be reviewed.

### Functional approval

- Confirm that the script correctly locates the project root.
- Confirm that the expected project folder structure is in place:

```text
Project2
├── Scripts
│   ├── 03_get_metadata_powershell.ps1
│   └── Logs
├── Video
│   └── active.avi
└── Reports
```

- Confirm that the metadata report is created successfully.
- Confirm that the log file is created when logging is enabled.
- Confirm that the `-NoLog` option behaves as expected.

### Technical approval

- Test the script in Windows PowerShell.
- Confirm behaviour in PowerShell 5.1 and/or PowerShell 7 if required.
- Confirm that COM access is permitted on the target machine.

### Data governance approval

- Confirm that the video file does not contain sensitive or restricted information.
- Confirm that file paths written to logs are acceptable for the project documentation.
- Confirm retention requirements for logs and generated metadata reports.

### Operational approval

- Confirm who is responsible for running the script.
- Confirm when the script should be run in the project workflow.
- Confirm whether logs should be retained with the project evidence bundle.
- Confirm that outputs are reviewed before being used in formal project documentation.

---

## 8. Suggested Classification

### Suggested classification

**Classification: Low-risk project automation script**

### Reasoning

The script is low risk because it:

- reads metadata from an existing local video file;
- creates text-based report and log files;
- does not modify the source video;
- does not send data outside the machine;
- does not require administrator privileges under normal conditions;
- does not execute external binaries;
- has clear error handling and logging.

### Suggested governance category

| Area | Classification |
|---|---|
| Risk level | Low |
| Data impact | Read-only access to source video; writes report and log files |
| Operational impact | Low |
| Security impact | Low, provided local file paths and logs are acceptable |
| Change control | Minor project utility script |
| Review requirement | Peer or supervisor review recommended before use in final evidence pack |
| Suitable for project evidence | Yes |

---

## 9. Recommended Approval Status

**Recommended status: Approved for controlled project use after syntax correction and test execution.**

The script is suitable for use as part of the AIML project evidence workflow once the syntax issues are corrected and a test run confirms that the expected metadata report and log file are generated.

Recommended approval condition:

> This script may be used to generate supporting metadata evidence for `active.avi`, provided that the output report and log file are reviewed and retained with the project documentation.

---

## 10. Final Summary

`03_get_metadata_powershell.ps1` is a Windows PowerShell automation script designed to extract selected Windows Explorer metadata from the project video file `active.avi`. It supports data traceability, repeatability, and documentation within the AIML-driven super-resolution and volumetric reconstruction project.

The script is appropriate as a low-risk project utility because it performs read-only inspection of the source video and creates controlled report and log outputs. It should be used on Windows systems and is best combined with the `ffprobe` metadata extraction script for a more complete technical record of the source video.
