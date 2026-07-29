# ============================================
# Script: 03_get_metadata_powershell.ps1
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Get metadata of active.avi using Windows PowerShell Shell.Application
#   - Extract selected Windows Explorer metadata fields
#   - Save metadata text file to Reports folder
#   - Save log to Logs folder beside script
#   - Create Reports and Logs folders if they do not exist
# ============================================

param (
    # Video file name inside the Video folder
    [string]$VideoFileName = "active.avi",

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
$VideoFile  = Join-Path -Path $ProjectRoot -ChildPath "Video\$VideoFileName"
$ReportsDir = Join-Path -Path $ProjectRoot -ChildPath "Reports"
$LogsDir    = Join-Path -Path $ScriptDir -ChildPath "Logs"

# Define output files
$OutputFile = Join-Path -Path $ReportsDir -ChildPath "03_active-avi_metadata_powershell.txt"
$LogFile    = Join-Path -Path $LogsDir -ChildPath "03_get_metadata_powershell.log"

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
    Write-Log "Script name: 03_get_metadata_powershell.ps1"
    Write-Log "Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"
    Write-Log "Script folder: $ScriptDir"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Video file: $VideoFile"
    Write-Log "Reports folder: $ReportsDir"

    if ($EnableLogging) {
        Write-Log "Logs folder: $LogsDir"
        Write-Log "Log file: $LogFile"
    }
    else {
        Write-Log "File logging disabled by -NoLog option." "WARNING"
    }

    # Check video file exists
    if (!(Test-Path -Path $VideoFile -PathType Leaf)) {
        throw "Video file not found: $VideoFile"
    }

    # Resolve full video path
    $FullPath = (Resolve-Path -Path $VideoFile).Path

    # Extract folder path and file name
    $FolderPath = Split-Path -Path $FullPath -Parent
    $FileName   = Split-Path -Path $FullPath -Leaf

    Write-Log "Resolved video path: $FullPath"
    Write-Log "Video folder path: $FolderPath"
    Write-Log "Video file name: $FileName"

    # Get video file information
    $VideoInfo = Get-Item -Path $FullPath

    Write-Log "Video file found."
    Write-Log "Video file size: $($VideoInfo.Length) bytes"
    Write-Log "Video last modified: $($VideoInfo.LastWriteTime)"

    # Metadata indices to extract from Windows Explorer details
    $MetadataIndexes = @(165, 192, 1, 2, 3, 4, 5, 9, 27, 29, 164, 198)

    Write-Log "Creating Shell.Application COM object."

    # Create Windows Shell COM object
    $Shell = New-Object -ComObject Shell.Application

    # Open folder namespace
    $Folder = $Shell.Namespace($FolderPath)

    if ($null -eq $Folder) {
        throw "Unable to access folder namespace: $FolderPath"
    }

    # Parse target file
    $File = $Folder.ParseName($FileName)

    if ($null -eq $File) {
        throw "Unable to parse video file using Shell.Application: $FileName"
    }

    Write-Log "Extracting selected Windows metadata fields."

    # Collect metadata
    $Metadata = foreach ($Index in $MetadataIndexes) {
        $Name  = $Folder.GetDetailsOf($null, $Index)
        $Value = $Folder.GetDetailsOf($File, $Index)

        if ([string]::IsNullOrWhiteSpace($Value)) {
            "{0}: {1} = {2}" -f $Index, $Name, $Value
        }
    }

    # If no metadata was returned, still create a useful report
    if ($null -eq $Metadata -or $Metadata.Count -eq 0) {
        Write-Log "No metadata values were returned for the selected indices." "WARNING"

        $Metadata = @(
            "No metadata values were returned for the selected Windows Explorer metadata indices."
            "File: $FullPath"
            "File size: $($VideoInfo.Length) bytes"
            "Last modified: $($VideoInfo.LastWriteTime)"
        )
    }

    # Add report header
    $ReportContent = @(
        "PowerShell Metadata Report"
        "Project: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"
        "Source file: $FullPath"
        "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        "File size: $($VideoInfo.Length) bytes"
        "Last modified: $($VideoInfo.LastWriteTime)"
        "============================================"
        ""
    ) + $Metadata

    # Save metadata report
    Set-Content -Path $OutputFile -Value $ReportContent -Encoding UTF8

    Write-Log "Metadata extraction completed successfully." "SUCCESS"
    Write-Log "Metadata entries saved: $($Metadata.Count)"
    Write-Log "Metadata report saved to: $OutputFile"
    Write-Log "Script completed successfully." "SUCCESS"
    Write-Log "============================================"
}
catch {
    Write-Log "Script failed: $($_.Exception.Message)" "ERROR"
    Write-Log "============================================"
    exit 1
}
finally {
    # Release COM objects where possible
    if ($null -ne $File) {
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($File) | Out-Null
    }

    if ($null -ne $Folder) {
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($Folder) | Out-Null
    }

    if ($null -ne $Shell) {
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($Shell) | Out-Null
    }

    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}