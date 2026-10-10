<#
.SYNOPSIS
    Generates a plain-text folder tree for a project and saves it as
    "<TopFolderName>-FolderStructure.txt".

.PARAMETER RootPath
    Folder to document. Defaults to one level above the script (project root, e.g. Project3 when the script is in Project3\Utils).

.PARAMETER OutputFolder
    Folder where the output file is saved. Defaults to "..\Utils" (relative to the script).

.PARAMETER ExcludeDotFolders
    Excludes any folder whose name starts with "." (e.g. .git, .venv-tomo, .vscode).

.PARAMETER IncludeFiles
    Lists files as well as folders.

.EXAMPLE
    .\ShowFolderTree.ps1
    .\ShowFolderTree.ps1 -ExcludeDotFolders
    .\ShowFolderTree.ps1 -ExcludeDotFolders -IncludeFiles
    .\ShowFolderTree.ps1 -RootPath "C:\Projects\Project2" -OutputFolder "C:\Temp"
#>

param(
    [string]$RootPath = (Join-Path $PSScriptRoot ".."),
    [string]$OutputFolder = (Join-Path $PSScriptRoot "..\Utils"),
    [switch]$ExcludeDotFolders,
    [switch]$IncludeFiles
)

function Show-FolderTree {
    param(
        [string]$Path,
        [string]$Indent = "",
        [ref]$Output,
        [bool]$ExcludeDot,
        [bool]$WithFiles
    )

    # Sub-folders
    $folders = Get-ChildItem -Path $Path -Directory -Force |
        Where-Object { -not ($ExcludeDot -and $_.Name.StartsWith(".")) } |
        Sort-Object Name

    # Files (only when requested)
    $files = @()
    if ($WithFiles) {
        $files = Get-ChildItem -Path $Path -File -Force |
            Where-Object { -not ($ExcludeDot -and $_.Name.StartsWith(".")) } |
            Sort-Object Name
    }

    # Folders first, then files
    foreach ($folder in $folders) {
        $Output.Value += "$Indent|-- $($folder.Name)"

        Show-FolderTree -Path $folder.FullName `
                        -Indent "$Indent    " `
                        -Output $Output `
                        -ExcludeDot $ExcludeDot `
                        -WithFiles $WithFiles
    }

    foreach ($file in $files) {
        $Output.Value += "$Indent|-- $($file.Name)"
    }
}

# Resolve the top folder
if (-not (Test-Path -Path $RootPath -PathType Container)) {
    Write-Error "Root path not found: $RootPath"
    exit 1
}
$RootFull   = (Resolve-Path -Path $RootPath).ProviderPath.TrimEnd('\', '/')
$TopFolder  = Split-Path -Path $RootFull -Leaf

# Output file: <TopFolder>-FolderStructure.txt
if (-not (Test-Path -Path $OutputFolder)) {
    New-Item -Path $OutputFolder -ItemType Directory -Force | Out-Null
}
$OutputFile = Join-Path (Resolve-Path -Path $OutputFolder).ProviderPath "$TopFolder-FolderStructure.txt"

# Build tree, starting with the top folder itself
$Tree = @($TopFolder)

Show-FolderTree -Path $RootFull `
                -Indent "    " `
                -Output ([ref]$Tree) `
                -ExcludeDot $ExcludeDotFolders.IsPresent `
                -WithFiles $IncludeFiles.IsPresent

# Save to file
$Tree | Set-Content -Path $OutputFile -Encoding UTF8

Write-Host "Top folder   : $TopFolder"
Write-Host "Exclude dot  : $($ExcludeDotFolders.IsPresent)"
Write-Host "Include files: $($IncludeFiles.IsPresent)"
Write-Host "Folder tree saved to $OutputFile"
