# Input AVI file
$aviFile = Join-Path $PSScriptRoot "..\Video\active.avi"

# Local FFmpeg executable
$ffmpeg = Join-Path $PSScriptRoot "..\Bin\ffmpeg.exe"

# Output folder: Images\origin
$outputDir = Join-Path $PSScriptRoot "..\Images\02_Frames"

# Create output folder if it does not exist
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Extract frames
& $ffmpeg -i $aviFile -vf "crop=940:600:0:40" "$outputDir\frame_%04d.png"

Write-Host "Frames extracted to:"
Write-Host $outputDir