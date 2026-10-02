# Script Description: ShowFolderTree.ps1

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `ShowFolderTree.ps1` |
| **Language** | PowerShell |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Project Utility / Documentation Automation |
| **Platform** | Windows PowerShell (or PowerShell Core, cross-platform) |
| **Execution Type** | Standalone utility script; run manually, no arguments required |

---

## Categories

1. **Project Structure Documentation**
   Recursively walks the project's folder hierarchy and records it as a plain-text tree, excluding known non-essential folders.

2. **Repeatable Documentation Automation**
   Provides a single, repeatable command to regenerate `Docs\FolderTree.txt` whenever the project's folder layout changes, keeping the recorded structure current without manual transcription.

---

## Script Purpose

`ShowFolderTree.ps1` generates a plain-text, indented representation of the project's directory structure and saves it to `Docs\FolderTree.txt`. It exists purely to keep the project's documented folder layout (referenced throughout the other script `.md` files and `FolderTree.txt`) accurate and easy to regenerate after restructuring.

### Key Features & Execution Flow

* **Recursive Folder Walk (`Show-FolderTree`):**
  * A recursive function that lists subfolders of a given path, indents child folders relative to their depth, and appends each line to an accumulating output collection passed by reference (`[ref]$Output`).
* **Exclusion Filter:**
  * Skips the `.venv-tomo` virtual environment folder so tool/dependency directories don't clutter the recorded tree.
* **Fixed Output Location:**
  * Always writes the result to `..\Docs\FolderTree.txt` (relative to the script's own folder), using UTF-8 encoding.
* **Two-Level Relative Root:**
  * Starts the walk from `..\..` (two levels above the script's folder — i.e. the project root), so the full project tree is captured, not just the `Scripts` folder it lives in.

### Step-by-Step Process Workflow

1. **Function Definition:**
   * Defines `Show-FolderTree`, which lists directories under `-Path`, sorts them alphabetically, excludes `.venv-tomo`, and recurses into each one with increased indentation.

2. **Tree Generation:**
   * Calls `Show-FolderTree -Path "..\.."` (the project root) with an empty output collection.

3. **File Export:**
   * Writes the accumulated tree lines to `..\Docs\FolderTree.txt` via `Set-Content -Encoding UTF8`.

4. **Confirmation Message:**
   * Prints the output file path to the console on completion.

---

## Business / Process Purpose

Clear, current documentation of the project's folder structure helps collaborators, reviewers, and the author's future self navigate the pipeline's inputs and outputs without needing to explore the filesystem manually.

`ShowFolderTree.ps1` supports this objective by:
1. **Reducing Documentation Drift:** Regenerating the folder tree from the live filesystem rather than relying on a document that can silently go stale as folders are added, renamed, or removed.
2. **Supporting Onboarding & Review:** Giving new collaborators or academic reviewers a quick, accurate map of where raw video, intermediate frames, cleaned outputs, datasets, and scripts live.
3. **Complementing Script-Level Documentation:** Providing the project-wide structural context that individual script `.md` files (which describe inputs/outputs relative to specific subfolders) assume as background.

---

## Command-Line Arguments & Parameters

`ShowFolderTree.ps1` takes no command-line arguments. Its behaviour is fixed by internal defaults:

| Internal Setting | Value | Description |
| :--- | :--- | :--- |
| Excluded folder name | `.venv-tomo` | Skipped during the recursive walk. |
| Root path walked | `..\..` (relative to script location) | Two levels above `Scripts\`, i.e. the project root. |
| Output file | `..\Docs\FolderTree.txt` | Fixed destination for the generated tree. |
| Output encoding | UTF-8 | Encoding used for `Set-Content`. |

---

## Input & Output Structure

```text
Project_Root/
├── Docs/
│   └── FolderTree.txt          # OUTPUT: generated folder tree listing
└── Scripts/
    └── ShowFolderTree.ps1      # Execution script
```

---

## Technical Dependencies

| Component | Purpose |
| :--- | :--- |
| `Get-ChildItem` (built-in) | Enumerates subdirectories at each recursion level. |
| `Set-Content` (built-in) | Writes the final tree text to `Docs\FolderTree.txt`. |

No external modules or third-party dependencies are required.

---

## Execution Examples

### Basic Execution (run from the Scripts folder)
```powershell
.\ShowFolderTree.ps1
```

### Typical Console Output
```text
Folder tree saved to ..\Docs\FolderTree.txt
```
