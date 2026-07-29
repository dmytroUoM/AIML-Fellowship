# ============================================
# Script: 02_get_metadata_ffprobe.ps1
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Get metadata of active.avi using ffprobe.exe
#   - Save JSON file to Reports folder
#   - Save log to Logs folder beside script
#   - Create Reports and Logs folders if they do not exist
#=============================================
# Part of folder scructure
# Project2
# ├── Bin3
# │	└── ffprobe.exe
# ├── Scripts
# │   ├── 02_get_metadata_ffprobe.ps1
# │   
# └── Logs
# │   └── 02_get_metadata_ffprobe.log
# ├── Video
# │   └── active.avi
# └── Reports    
#     └── 02_avi_metadata_ffprobe.json
# ============================================

param (
    # Default video file name
    [string]$VideoFileName = "active.avi",

    # Use -NoLog to disable log file creation
    [switch]$NoLog
)

$ErrorActionPreference = "Stop"

# Enable logging unless -NoLog is used
$EnableLogging = -not $NoLog

# Get the folder where this script is located
$ScriptDir = $PSScriptRoot

# If script is run in older PowerShell versions where $PSScriptRoot may be empty
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
}

# Project root is one level above the script folder
$ProjectRoot = Split-Path -Path $ScriptDir -Parent

# Define paths
$FfprobeExe = Join-Path -Path $ProjectRoot -ChildPath "Bin\ffprobe.exe"
$VideoFile  = Join-Path -Path $ProjectRoot -ChildPath "Video\$VideoFileName"
$ReportsDir = Join-Path -Path $ProjectRoot -ChildPath "Reports"
$LogsDir    = Join-Path -Path $ScriptDir -ChildPath "Logs"

# Define output files
$MetadataFile = Join-Path -Path $ReportsDir -ChildPath "02_avi_metadata_ffprobe.json"
$LogFile      = Join-Path -Path $LogsDir -ChildPath "02_get_metadata_ffprobe.log"

# Create Reports folder if it does not exist
if (!(Test-Path -Path $ReportsDir)) {
    New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
}

# Create Logs folder only if logging is enabled
if ($EnableLogging -and !(Test-Path -Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Function to write log messages
function Write-Log {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Message,

        [ValidateSet("INFO", "WARNING", "ERROR", "SUCCESS")]
        [string]$Level = "INFO"
    )

    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"

    Write-Host $LogEntry

    if ($EnableLogging) {
        Add-Content -Path $LogFile -Value $LogEntry -Encoding UTF8
    }
}

try {
    Write-Log "============================================"
    Write-Log "Script started."
    Write-Log "Script name: 02_get_metadata_ffprobe.ps1"
    Write-Log "Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"
    Write-Log "Script folder: $ScriptDir"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "ffprobe path: $FfprobeExe"
    Write-Log "Video file: $VideoFile"
    Write-Log "Reports folder: $ReportsDir"

    if ($EnableLogging) {
        Write-Log "Logs folder: $LogsDir"
        Write-Log "Log file: $LogFile"
    }
    else {
        Write-Log "Logging to file is disabled."
    }

    # Check ffprobe.exe exists
    if (!(Test-Path -Path $FfprobeExe -PathType Leaf)) {
        throw "ffprobe.exe not found. Expected location: $FfprobeExe"
    }

    # Check video file exists
    if (!(Test-Path -Path $VideoFile -PathType Leaf)) {
        throw "Video file not found: $VideoFile"
    }

    # Get video file information
    $VideoInfo = Get-Item -Path $VideoFile

    Write-Log "Video file found."
    Write-Log "Video file size: $($VideoInfo.Length) bytes"
    Write-Log "Video last modified: $($VideoInfo.LastWriteTime)"

    # Temporary error file for ffprobe stderr
    $FfprobeErrorFile = Join-Path -Path $env:TEMP -ChildPath "ffprobe_error_$((Get-Date).ToString('yyyyMMdd_HHmmss')).txt"

    Write-Log "Running ffprobe metadata extraction..."

    # Run ffprobe and capture JSON output
    $MetadataJson = & $FfprobeExe `
        -v quiet `
        -print_format json `
        -show_format `
        -show_streams `
        $VideoFile `
        2> $FfprobeErrorFile

    $ExitCode = $LASTEXITCODE

    # Check ffprobe exit code
    if ($ExitCode -ne 0) {
        $FfprobeError = ""

        if (Test-Path -Path $FfprobeErrorFile) {
            $FfprobeError = Get-Content -Path $FfprobeErrorFile -Raw
        }

        throw "ffprobe failed with exit code $ExitCode. Error: $FfprobeError"
    }

    # Check metadata was returned
    if ([string]::IsNullOrWhiteSpace($MetadataJson)) {
        throw "ffprobe returned empty metadata output."
    }

    # Validate that output is proper JSON
    try {
        $null = $MetadataJson | ConvertFrom-Json
        Write-Log "ffprobe output validated as JSON." "SUCCESS"
    }
    catch {
        throw "ffprobe output is not valid JSON. $($_.Exception.Message)"
    }

    # Save JSON metadata file
    Set-Content -Path $MetadataFile -Value $MetadataJson -Encoding UTF8

    Write-Log "Metadata extracted successfully." "SUCCESS"
    Write-Log "Metadata saved to: $MetadataFile"
    Write-Log "Script completed successfully." "SUCCESS"
    Write-Log "============================================"

    # Remove temporary error file if it exists
    if (Test-Path -Path $FfprobeErrorFile) {
        Remove-Item -Path $FfprobeErrorFile -Force
    }
}
catch {
    Write-Log "Script failed: $($_.Exception.Message)" "ERROR"
    Write-Log "============================================"
    exit 1
}
