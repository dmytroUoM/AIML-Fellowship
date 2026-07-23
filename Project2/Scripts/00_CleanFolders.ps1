$folders = @(
    "..\Images",
    "..\Reports"
)

foreach ($folder in $folders) {
    if (Test-Path $folder) {
        Get-ChildItem -Path $folder -Force |
            Remove-Item -Recurse -Force

        Write-Host "Cleaned: $folder"
    }
    else {
        Write-Warning "Folder not found: $folder"
    }
}