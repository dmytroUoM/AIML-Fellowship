
# Create Reports folder if it doesn't exist
if (!(Test-Path "..\Reports")) {
    New-Item -ItemType Directory -Path "..\Reports" | Out-Null
}

# Calculate SHA256 hash and save only the hash value
(Get-FileHash "..\origin\try.avi" -Algorithm SHA256).Hash |
    Set-Content "..\Reports\01_avi_hash.txt"
