# Input AVI file
$aviFile = Join-Path $PSScriptRoot "..\Origin\try.avi"

# Local FFmpeg executable
$ffmpeg = Join-Path $PSScriptRoot "..\Bin\ffmpeg.exe"

# Output folder: frames\origin
$outputDir = Join-Path $PSScriptRoot "..\Frames\Cropped"

# Create output folder if it does not exist
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Extract frames
& $ffmpeg -i $aviFile -vf "crop=430:570:9:109" "$outputDir\frame_%04d.png"

Write-Host "Frames extracted to:"
Write-Host $outputDir