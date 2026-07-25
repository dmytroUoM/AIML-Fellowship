function Show-FolderTree {
    param(
        [string]$Path = "..",
        [string]$Indent = "",
        [ref]$Output
    )

    $folders = Get-ChildItem -Path $Path -Directory |
        Where-Object { $_.Name -ne ".venv-tomo" } |
        Sort-Object Name

    foreach ($folder in $folders) {
        $line = "$Indent|-- $($folder.Name)"
        $Output.Value += $line

        Show-FolderTree -Path $folder.FullName `
                        -Indent "$Indent    " `
                        -Output $Output
    }
}

# Output file
$OutputFile = "..\Docs\FolderTree.txt"

# Create output collection
$Tree = @()

# Generate tree
Show-FolderTree -Path "..\.." -Output ([ref]$Tree)

# Save to file
$Tree | Set-Content -Path $OutputFile -Encoding UTF8

Write-Host "Folder tree saved to $OutputFile"