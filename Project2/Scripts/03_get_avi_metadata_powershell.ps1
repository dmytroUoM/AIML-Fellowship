# Paths relative to the script location

$filePath   = Join-Path $PSScriptRoot "..\Video\active.avi"
$outputFile = Join-Path $PSScriptRoot "..\Reports\03_active-avi_metadata_powershell.txt"

$fullPath = (Resolve-Path $filePath).Path

$folderPath = Split-Path $fullPath -Parent
$fileName   = Split-Path $fullPath -Leaf

$shell = New-Object -ComObject Shell.Application
$folder = $shell.Namespace($folderPath)
$file = $folder.ParseName($fileName)

$metadata = foreach ($index in 165,192,1,2,3,4,5,9,27,29,164,198) {
    $name = $folder.GetDetailsOf($null, $index)
    $value = $folder.GetDetailsOf($file, $index)

    if ($value) {
        "{0}: {1} = {2}" -f $index, $name, $value
    }
}

$metadata | Set-Content -Path $outputFile -Encoding UTF8

Write-Host "Metadata saved to $outputFile"