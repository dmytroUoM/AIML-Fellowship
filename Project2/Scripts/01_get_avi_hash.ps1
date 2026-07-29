# ============================================
# Script: 01_Get_Avi_SHA256_Hash.ps1
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose: Calculate SHA256 hash of active.avi
#          Save hash to Reports folder
#          Save log to Logs folder beside script
# ============================================

$ErrorActionPreference = "Stop"

# Get the folder where this script is located
$ScriptDir = $PSScriptRoot

# Project root is one level above the script folder
$ProjectRoot = Split-Path $ScriptDir -Parent

# Define input and output paths
$VideoFile  = Join-Path $ProjectRoot "Video\active.avi"
$ReportsDir = Join-Path $ProjectRoot "Reports"
$LogsDir    = Join-Path $ScriptDir "Logs"

$HashFile = Join-Path $ReportsDir "01_avi_hash.txt"
$LogFile  = Join-Path $LogsDir "01_avi_hash.log"

# Create Reports folder if it does not exist
if (!(Test-Path $ReportsDir)) {
    New-Item -ItemType Directory -Path $ReportsDir | Out-Null
}

# Create Logs folder if it does not exist
if (!(Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

# Function to write log messages
function Write-Log {
    param (
        [string]$Message,
        [string]$Level = "INFO"
    )

    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"

    Add-Content -Path $LogFile -Value $LogEntry
    Write-Host $LogEntry
}

try {
    Write-Log "Script started."
    Write-Log "Script folder: $ScriptDir"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Video file: $VideoFile"
    Write-Log "Reports folder: $ReportsDir"
    Write-Log "Logs folder: $LogsDir"

    # Check that video file exists
    if (!(Test-Path $VideoFile)) {
        throw "Video file not found: $VideoFile"
    }

    Write-Log "Calculating SHA256 hash."

    # Calculate SHA256 hash
    $Hash = (Get-FileHash -Path $VideoFile -Algorithm SHA256).Hash

    # Save only the hash value
    Set-Content -Path $HashFile -Value $Hash

    Write-Log "SHA256 hash calculated successfully."
    Write-Log "Hash saved to: $HashFile"
    Write-Log "Hash value: $Hash"
    Write-Log "Script completed successfully."
}
catch {
    Write-Log "Script failed: $($_.Exception.Message)" "ERROR"
    exit 1
}
