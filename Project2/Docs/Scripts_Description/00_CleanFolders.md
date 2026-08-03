# Script Description: 00_CleanFolders.ps1

## Script Purpose
This PowerShell script safely deletes all files and subfolders inside a predefined, approved list of output directories (`..\Images` and `..\Reports`), while preserving the top-level folders themselves. It is designed to reset "scratch" or "output" folders to an empty state before a new pipeline run, without risking accidental deletion of unintended locations. All actions are written to a persistent log file, and a `-WhatIfMode` switch allows a dry-run preview of what would be deleted before any destructive action is taken.

## Business / Process Purpose
In a repeatable data or ML pipeline (e.g., a project that regenerates images and reports on each run), stale output files can accumulate and cause confusion, disk bloat, or false positives when validating new results. This script supports **repeatability and cleanliness of pipeline runs** by providing a controlled, auditable way to clear known output locations before regeneration — for example, as Step 00 in a numbered script sequence (`00_CleanFolders.ps1`, `01_...`, `02_...`) that resets the environment ahead of downstream steps.

## Log the Result
- The script auto-creates a `Logs` subfolder next to itself (`<ScriptRoot>\Logs\CleanFolders.log`) if it doesn't already exist.
- Every run appends timestamped entries (`yyyy-MM-dd HH:mm:ss`) with severity levels: `INFO`, `WARNING`, `ERROR`.
- Logged events include: script start/end, WhatIf mode notice, each folder processed, resolved full path, item counts found, each item deleted (or would-be-deleted under WhatIf), and any errors/safety-check failures.
- Console output mirrors the log (color-coded: red for errors, yellow warnings for `Write-Warning`), giving both a real-time view and a persistent audit trail.
- The log file path is echoed at the end of every run for easy retrieval.

## Technology Assessment
| Aspect | Detail |
|---|---|
| **Language/Runtime** | Windows PowerShell (5.1) / PowerShell 7+ compatible |
| **Dependencies** | None beyond built-in cmdlets (`Get-ChildItem`, `Remove-Item`, `Resolve-Path`, `Test-Path`, `.NET` `[string]`/`[System.IO.Path]` methods) |
| **Portability** | Windows-oriented (relies on drive-letter path roots); would need adaptation for cross-platform PowerShell on Linux/macOS |
| **Error Handling** | Try/catch per folder iteration; failures in one folder do not halt processing of the others |
| **Idempotency** | Yes — running repeatedly on an already-empty folder is a safe no-op |
| **Maintainability** | Straightforward, well-commented, single-file script; folder list and approval list are simple arrays at the top, easy to extend |

## Approval Considerations
- **Destructive by design**: default behavior (without `-WhatIfMode`) permanently deletes files and folders using `Remove-Item -Recurse -Force`, with no recycle-bin recovery.
- **Scope-limiting safeguards already present**, which should be reviewed and confirmed as sufficient by an approver:
  1. Folder must exist (`Test-Path`) before any action.
  2. Path is resolved to a full path and checked for null/blank.
  3. Folder **name** (leaf) must match an explicit approved allow-list (`Images`, `Reports`) — this is the primary control preventing scope creep.
  4. Refuses to operate if the resolved path equals a drive root (e.g., `C:\`).
- **Recommended additional review points before production approval**:
  - Confirm the relative paths (`..\Images`, `..\Reports`) always resolve correctly regardless of the working directory the script is launched from (currently relies on the caller's current directory, not `$scriptRoot`, for the initial `Test-Path`/`Resolve-Path` calls — worth validating or hardening to use absolute paths derived from `$scriptRoot`).
  - Confirm log folder/file does not grow unbounded (no rotation/retention policy is implemented).
  - Confirm this script will be run only by authorized users/service accounts, since no execution-time authentication/authorization is built in beyond OS-level permissions.
  - Confirm `-WhatIfMode` is used first in any new environment before a live run.

## Suggested Classification
**Category:** Operational / Infrastructure Utility Script — File & Folder Maintenance
**Risk Level:** Medium (destructive file operations, mitigated by allow-list, WhatIf mode, and logging)
**Recommended Handling:** Treat as a controlled operational script requiring change review before modification (particularly the `$folders` / `$approvedFolderNames` arrays), with mandatory `-WhatIfMode` validation step in any new deployment context.
