# ============================================
# Script: 04_extract_origin_frames.ps1
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Extract original frames from active.avi using ffmpeg.exe
#   - Save extracted frames to Images\01_Origin folder
#   - Save log to Logs folder beside script
#   - Create Images\01_Origin and Logs folders if they do not exist
# ============================================

param (
    # Video file name inside the Video folder
    [string]$VideoFileName = "active.avi",

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

# Fallback for older PowerShell versions
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
}

# Project root is one level above the Scripts folder
$ProjectRoot = Split-Path -Path $ScriptDir -Parent

# Define paths
$AviFile   = Join-Path -Path $ProjectRoot -ChildPath "Video\$VideoFileName"
$FfmpegExe = Join-Path -Path $ProjectRoot -ChildPath "Bin\ffmpeg.exe"
$OutputDir = Join-Path -Path $ProjectRoot -ChildPath "Images\01_Origin"
$LogsDir   = Join-Path -Path $ScriptDir -ChildPath "Logs"

# Define output files
$LogFile = Join-Path -Path $LogsDir -ChildPath "04_extract_origin_frames.log"

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
    Write-Log "Script name: 04_extract_origin_frames.ps1"
    Write-Log "Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"
    Write-Log "Script folder: $ScriptDir"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Input video file: $AviFile"
    Write-Log "FFmpeg path: $FfmpegExe"
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

    # Check video file exists
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

    # Count existing frames before extraction
    $ExistingFramesBefore = @(Get-ChildItem -Path $OutputDir -Filter "*.png" -File -ErrorAction SilentlyContinue).Count
    Write-Log "Existing PNG frames before extraction: $ExistingFramesBefore"

    # Temporary file for ffmpeg error output
    $FfmpegErrorFile = Join-Path -Path $env:TEMP -ChildPath "ffmpeg_extract_frames_$((Get-Date).ToString('yyyyMMdd_HHmmss')).txt"

    Write-Log "Starting frame extraction using ffmpeg."

    # FFmpeg writes normal version/progress info to stderr even on success.
    # PowerShell treats stderr lines from native commands as error-stream
    # records, and with $ErrorActionPreference = "Stop" the very first such
    # line (ffmpeg's version banner) would otherwise abort the script as if
    # it were a real failure - before $LASTEXITCODE is ever even checked.
    # Temporarily relax ErrorActionPreference so stderr is simply redirected
    # to the file as intended, then restore it and rely on $LASTEXITCODE
    # (checked below) to detect real failures.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # Extract frames from video
    & $FfmpegExe `
        -y `
        -i $AviFile `
        $OutputPattern `
        2> $FfmpegErrorFile

    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    Write-Log "FFmpeg exit code: $ExitCode"

    # Check ffmpeg exit code
    if ($ExitCode -ne 0) {
        $FfmpegError = ""

        if (Test-Path -Path $FfmpegErrorFile) {
            $FfmpegError = Get-Content -Path $FfmpegErrorFile -Raw
        }

        throw "ffmpeg failed with exit code $ExitCode. Error: $FfmpegError"
    }

    # Count frames after extraction
    $ExtractedFrames = @(Get-ChildItem -Path $OutputDir -Filter "*.png" -File -ErrorAction SilentlyContinue).Count
    $NewFrames = $ExtractedFrames - $ExistingFramesBefore

    Write-Log "Frame extraction completed successfully." "SUCCESS"
    Write-Log "Total PNG frames currently in output folder: $ExtractedFrames"
    Write-Log "Approximate new frames created during this run: $NewFrames"
    Write-Log "Frames extracted to: $OutputDir"

    # Optional FFmpeg diagnostic note (matches 05's pattern: capture but
    # don't treat as an error when exit code was 0)
    if (Test-Path -Path $FfmpegErrorFile) {
        $FfmpegErrorText = Get-Content -Path $FfmpegErrorFile -Raw

        if (-not [string]::IsNullOrWhiteSpace($FfmpegErrorText)) {
            Write-Log "FFmpeg diagnostic output was captured. This is normal and was not treated as an error because exit code was 0."
        }

        Remove-Item -Path $FfmpegErrorFile -Force
    }

    Write-Log "Script completed successfully." "SUCCESS"
    Write-Log "============================================"
}
catch {
    Write-Log "Script failed: $($_.Exception.Message)" "ERROR"
    Write-Log "============================================"
    exit 1
}