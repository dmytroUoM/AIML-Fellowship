# ============================================
# Script: 05_extract_crop_frames_from_avi.ps1
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Extract frames from active.avi using ffmpeg.exe
#   - Crop each frame using crop=940:600:0:40
#   - Save cropped frames to Images\02_Frames folder
#   - Save log to Logs folder beside script
#   - Create Images\02_Franes  and Logs folders if they do not exist
# ============================================

param (
    # Video file name inside the Video folder
    [string]$VideoFileName = "active.avi",

    # FFmpeg crop filter
    [string]$CropFilter = "crop=940:600:0:40",

    # Output frame image pattern
    [string]$FramePattern = "frame_%04d.png",

    # Use -NoLog to disable saving the log file
    [switch]$NoLog
)

$ErrorActionPreference = "Stop"

# Enable logging unless -NoLog is used
$EnableLogging = -not $NoLog

# Get the folder where this script is located
$ScriptDir = $PSScriptRoot

# Fallback for older PowerShell versions or unusual execution contexts
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
}

# Project root is one level above the Scripts folder
$ProjectRoot = Split-Path -Path $ScriptDir -Parent

# Define paths
$AviFile   = Join-Path -Path $ProjectRoot -ChildPath "Video\$VideoFileName"
$FfmpegExe = Join-Path -Path $ProjectRoot -ChildPath "Bin\ffmpeg.exe"
$OutputDir = Join-Path -Path $ProjectRoot -ChildPath "Images\02_Frames"
$LogsDir   = Join-Path -Path $ScriptDir -ChildPath "Logs"

# Define output files
$LogFile = Join-Path -Path $LogsDir -ChildPath "05_extract_crop_frames_from_avi.log"

# Create output folder if it does not exist
if (!(Test-Path -Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
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
    Write-Log "Script name: 05_extract_crop_frames_from_avi.ps1"
    Write-Log "Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"
    Write-Log "Script folder: $ScriptDir"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Input video file: $AviFile"
    Write-Log "FFmpeg path: $FfmpegExe"
    Write-Log "Crop filter: $CropFilter"
    Write-Log "Output folder: $OutputDir"

    if ($EnableLogging) {
        Write-Log "Logs folder: $LogsDir"
        Write-Log "Log file: $LogFile"
    }
    else {
        Write-Log "File logging disabled by -NoLog option." "WARNING"
    }

    # Check ffmpeg.exe exists
    if (!(Test-Path -Path $FfmpegExe -PathType Leaf)) {
        throw "ffmpeg.exe not found. Expected location: $FfmpegExe"
    }

    # Check input video exists
    if (!(Test-Path -Path $AviFile -PathType Leaf)) {
        throw "Input video file not found: $AviFile"
    }

    # Get video file information
    $VideoInfo = Get-Item -Path $AviFile

    Write-Log "Input video file found."
    Write-Log "Video file size: $($VideoInfo.Length) bytes"
    Write-Log "Video last modified: $($VideoInfo.LastWriteTime)"

    # Define output frame path pattern
    $OutputPattern = Join-Path -Path $OutputDir -ChildPath $FramePattern

    Write-Log "Frame output pattern: $OutputPattern"

    # Temporary file for FFmpeg diagnostic output
    # FFmpeg writes normal progress/version information to stderr,
    # so stderr output must not automatically be treated as an error.
    $FfmpegOutputFile = Join-Path -Path $env:TEMP -ChildPath "ffmpeg_crop_frames_$((Get-Date).ToString('yyyyMMdd_HHmmss')).txt"

    Write-Log "Starting cropped frame extraction using FFmpeg."

    # FFmpeg writes normal version/progress info to stderr even on success.
    # PowerShell treats stderr lines from native commands as error-stream
    # records, and with $ErrorActionPreference = "Stop" the very first such
    # line would otherwise abort the script as if it were a real failure.
    # Temporarily relax ErrorActionPreference so stderr is simply redirected
    # to the file as intended, then restore it and rely on $LASTEXITCODE
    # (checked below) to detect real failures.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # Run FFmpeg frame extraction and crop
    & $FfmpegExe `
        -y `
        -i $AviFile `
        -vf $CropFilter `
        $OutputPattern `
        2> $FfmpegOutputFile

    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    Write-Log "FFmpeg exit code: $ExitCode"

    # Only treat FFmpeg as failed if exit code is not zero
    if ($ExitCode -ne 0) {
        $FfmpegOutput = ""

        if (Test-Path -Path $FfmpegOutputFile) {
            $FfmpegOutput = Get-Content -Path $FfmpegOutputFile -Raw
        }

        throw "FFmpeg failed with exit code $ExitCode. Details: $FfmpegOutput"
    }

    # Count frames after extraction
    $NewFrames = @(Get-ChildItem -Path $OutputDir -Filter "frame_*.png" -File -ErrorAction SilentlyContinue).Count

    Write-Log "Cropped frame extraction completed successfully." "SUCCESS"
    Write-Log "New frames created during this run: $NewFrames"
    Write-Log "Frames extracted to: $OutputDir" "SUCCESS"

    # Optional FFmpeg diagnostic note
    if (Test-Path -Path $FfmpegOutputFile) {
        $FfmpegOutputText = Get-Content -Path $FfmpegOutputFile -Raw

        if (-not [string]::IsNullOrWhiteSpace($FfmpegOutputText)) {
            Write-Log "FFmpeg diagnostic output was captured. This is normal and was not treated as an error because exit code was 0."
        }

        Remove-Item -Path $FfmpegOutputFile -Force
    }

    Write-Log "Script completed successfully." "SUCCESS"
    Write-Log "============================================"
}
catch {
    Write-Log "Script failed: $($_.Exception.Message)" "ERROR"
    Write-Log "============================================"
    exit 1
}