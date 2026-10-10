# =====================================================
# Script: Create-SampleImages.ps1
#
# Purpose:
#   Create GitHub sample images for Project2 and Project3.
#
# Behaviour:
#   - Scans Project2 and Project3
#   - Searches only within Images folders
#   - Finds PNG files containing "0001"
#   - Creates:
#         sample.png
#         sample_mask.png
#   - Does not overwrite existing samples
#
# Examples matched:
#   frame_0001.png
#   frame_0001_mask.png
#   frame_0001_dark_00.png
#   anything_0001_anything.png
# =====================================================

$RootFolder = Get-Location

$Projects = @(
    "Project2",
    "Project3"
)

$Created = 0
$Skipped = 0

foreach ($Project in $Projects)
{
    $ProjectPath = Join-Path $RootFolder $Project

    if (!(Test-Path $ProjectPath))
    {
        Write-Host "[WARNING] Missing project: $Project"
        continue
    }

    Write-Host ""
    Write-Host "Scanning $Project ..."
    Write-Host ""

    $ImageFolders = Get-ChildItem `
        -Path $ProjectPath `
        -Directory `
        -Recurse |
        Where-Object {
            $_.FullName -match "\\Images\\"
        }

    foreach ($Folder in $ImageFolders)
    {
        $Candidates = Get-ChildItem `
            -Path $Folder.FullName `
            -File `
            -Filter *.png |
            Where-Object {
                $_.Name -match "0001"
            }

        if ($Candidates.Count -eq 0)
        {
            continue
        }

        foreach ($Candidate in $Candidates)
        {
            if ($Candidate.Name -match "_mask")
            {
                $TargetFile = Join-Path `
                    $Folder.FullName `
                    "sample_mask.png"
            }
            else
            {
                $TargetFile = Join-Path `
                    $Folder.FullName `
                    "sample.png"
            }

            if (Test-Path $TargetFile)
            {
                $Skipped++
                continue
            }

            Copy-Item `
                $Candidate.FullName `
                $TargetFile

            $Created++

            Write-Host "[OK] Created:"
            Write-Host "     $TargetFile"
            Write-Host "     From: $($Candidate.Name)"
        }
    }
}

Write-Host ""
Write-Host "==========================================="
Write-Host "Sample generation completed"
Write-Host "Created : $Created"
Write-Host "Skipped : $Skipped"
Write-Host "==========================================="