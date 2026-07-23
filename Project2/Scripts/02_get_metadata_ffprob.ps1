
# Create Reports folder if it doesn't exist
if (!(Test-Path "..\Reports")) {
    New-Item -ItemType Directory -Path "..\Reports" | Out-Null
}

# Run ffprobe and save JSON output
& "..\Bin\ffprobe.exe" `
    -v quiet `
    -print_format json `
    -show_format `
    -show_streams `
    "..\Video\active.avi" |
    Set-Content "..\Reports\02_avi_metadata_ffprobe.json"
