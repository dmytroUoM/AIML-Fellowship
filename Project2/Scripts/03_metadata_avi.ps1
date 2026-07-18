
param(
    [Parameter(Mandatory = $true)]
    [string]$filePath
)

$fullPath = (Resolve-Path $filePath).Path

$folderPath = Split-Path $fullPath -Parent
$fileName   = Split-Path $fullPath -Leaf

Write-Host "Folder:" $folderPath
Write-Host "File:" $fileName

$shell = New-Object -ComObject Shell.Application
$folder = $shell.Namespace($folderPath)

if ($null -eq $folder) {
    Write-Host "ERROR: Folder not found by Shell.Application"
    exit
}

$file = $folder.ParseName($fileName)

if ($null -eq $file) {
    Write-Host "ERROR: File not found by Shell.Application"
    exit
}

1,2,3,4,5,9,27,29,164,165,192,198 | ForEach-Object {
    $name = $folder.GetDetailsOf($null, $_)
    $value = $folder.GetDetailsOf($file, $_)

    if ($value) {
        "{0}: {1} = {2}" -f $_, $name, $value
    }
}
