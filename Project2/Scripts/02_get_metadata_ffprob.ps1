
# Create Reports folder if it doesn't exist
if (!(Test-Path "..\Reports")) {
    New-Item -ItemType Directory -Path "..\Reports" | Out-Null
}

# Run ffprobe and save JSON output
& "..\bin\ffprobe.exe" `
    -v quiet `
    -print_format json `
    -show_format `
    -show_streams `
    "..\origin\active.avi" |
    Set-Content "..\Reports\02_avi_metadata_ffprobe.json"
