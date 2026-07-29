<#
.SYNOPSIS
    Safely cleans the contents of approved output folders.

.DESCRIPTION
    This script removes all files and subfolders inside predefined output folders,
    while preserving the folders themselves. It logs all actions to a log file.

.NOTES
    Intended folders:
    - ..\Images
    - ..\Reports
	
.EXECUTION
	.\00_CleanFolders.ps1
	.\00_CleanFolders.ps1 -WhatIfMode

#>

param(
    [switch]$WhatIfMode
)

# Folders to clean
$folders = @(
    "..\Images",
    "..\Reports"
)

# Approved folder names only
$approvedFolderNames = @(
    "Images",
    "Reports"
)

# Log setup
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logFolder = Join-Path $scriptRoot "Logs"
$logFile = Join-Path $logFolder "CleanFolders.log"

# Ensure log folder exists
if (-not (Test-Path $logFolder)) {
    New-Item -Path $logFolder -ItemType Directory -Force | Out-Null
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "$timestamp [$Level] $Message"

    Add-Content -Path $logFile -Value $entry

    switch ($Level) {
        "ERROR" {
            Write-Host "ERROR: $Message" -ForegroundColor Red
        }
        "WARNING" {
            Write-Warning $Message
        }
        default {
            Write-Host $Message
        }
    }
}

Write-Log "--------------------------------------------------"
Write-Log "Starting folder cleanup script."
Write-Log "Script location: $scriptRoot"

if ($WhatIfMode) {
    Write-Log "Running in WHATIF mode. No files will be deleted." "WARNING"
}

foreach ($folder in $folders) {

    try {
        Write-Log "Processing folder reference: $folder"

        # Check folder exists
        if (-not (Test-Path $folder)) {
            Write-Log "Folder not found: $folder" "WARNING"
            continue
        }

        # Resolve to full path
        $resolvedPath = Resolve-Path $folder -ErrorAction Stop
        $fullPath = $resolvedPath.Path
        $folderName = Split-Path $fullPath -Leaf

        Write-Log "Resolved path: $fullPath"

        # Safety check 1: prevent empty or invalid paths
        if ([string]::IsNullOrWhiteSpace($fullPath)) {
            Write-Log "Resolved folder path is empty or invalid." "ERROR"
            continue
        }

        # Safety check 2: folder must be approved by name
        if ($approvedFolderNames -notcontains $folderName) {
            Write-Log "Folder is not in approved folder list: $fullPath" "ERROR"
            continue
        }

        # Safety check 3: prevent root drive cleanup
        $rootPath = [System.IO.Path]::GetPathRoot($fullPath)

        if ($fullPath -eq $rootPath) {
            Write-Log "Refusing to clean root path: $fullPath" "ERROR"
            continue
        }

        # Get contents
        $items = Get-ChildItem -Path $fullPath -Force -ErrorAction Stop

        if ($items.Count -eq 0) {
            Write-Log "Folder already empty: $fullPath"
            continue
        }

        Write-Log "Items found for deletion in $fullPath : $($items.Count)"

        foreach ($item in $items) {
            if ($WhatIfMode) {
                Write-Log "WHATIF: Would delete $($item.FullName)" "WARNING"
            }
            else {
                Write-Log "Deleting: $($item.FullName)"
                Remove-Item -Path $item.FullName -Recurse -Force -ErrorAction Stop
            }
        }

        if ($WhatIfMode) {
            Write-Log "WHATIF: Cleanup simulated for folder: $fullPath" "WARNING"
        }
        else {
            Write-Log "Successfully cleaned folder: $fullPath"
        }
    }
    catch {
        Write-Log "Failed to clean folder '$folder'. Error: $($_.Exception.Message)" "ERROR"
    }
}

Write-Log "Folder cleanup script completed."
Write-Log "Log file location: $logFile"