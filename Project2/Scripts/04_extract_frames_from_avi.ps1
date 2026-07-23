# Input AVI file
$aviFile = Join-Path $PSScriptRoot "..\Video\active.avi"

# Local FFmpeg executable
$ffmpeg = Join-Path $PSScriptRoot "..\Bin\ffmpeg.exe"

# Output folder: frames\origin
$outputDir = Join-Path $PSScriptRoot "..\Images\01_Origin"

# Create output folder if it does not exist
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Extract frames
& $ffmpeg -i $aviFile "$outputDir\frame_%04d.png"

Write-Host "Frames extracted to:"
Write-Host $outputDir