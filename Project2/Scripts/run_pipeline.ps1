# ============================================
# Script: run_pipeline.ps1
# Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks
# Purpose:
#   - Convenient menu-driven entry point to run the project's script
#     sequences (legacy core, legacy visual pipeline, AIML data/training
#     branch) without needing to remember or type out each command by hand.
#   - This is a THIN WRAPPER: it calls the existing scripts with their
#     existing parameters in the correct order. It does not duplicate any
#     of their logic. Each underlying script's own log file is still the
#     authoritative record of what happened during that step.
#   - Stops and asks whether to continue if any step fails (non-zero exit
#     code), rather than silently running the rest of a broken sequence.
#
# Usage:
#   Run from the Scripts folder:
#     .\run_pipeline.ps1
#   You may need to allow script execution first (once, as Administrator
#   or for your user):
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# Note: this is a CLI menu (Phase 1). A friendlier UI (e.g. a simple
# desktop or web front-end) is intended as a later deployment phase once
# the pipeline itself is stable - see notes at the bottom of this file.
# ============================================

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
}

# ----------------------------------------------------
# Detect Python launcher: prefer 'py' (matches this project's convention),
# fall back to 'python' if 'py' isn't on PATH.
# ----------------------------------------------------
function Get-PythonLauncher {
    if (Get-Command py -ErrorAction SilentlyContinue) { return "py" }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "Neither 'py' nor 'python' was found on PATH. Activate your virtual environment first."
}
$PythonExe = Get-PythonLauncher

# ----------------------------------------------------
# Run one step (a .ps1 or .py script), report pass/fail clearly, and offer
# to stop the whole sequence if a step fails.
# ----------------------------------------------------
function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )

    $fullPath = Join-Path -Path $ScriptDir -ChildPath $ScriptName
    if (!(Test-Path -Path $fullPath)) {
        Write-Host "[SKIP] $ScriptName not found at $fullPath" -ForegroundColor Yellow
        return $false
    }

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Running: $ScriptName $($Arguments -join ' ')" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan

    $stepStart = Get-Date
    $exitCode = 0

    try {
        if ($ScriptName -like "*.ps1") {
            & $fullPath @Arguments
            $exitCode = $LASTEXITCODE
        }
        elseif ($ScriptName -like "*.py") {
            & $PythonExe $fullPath @Arguments
            $exitCode = $LASTEXITCODE
        }
        else {
            Write-Host "[SKIP] Unrecognized script type: $ScriptName" -ForegroundColor Yellow
            return $false
        }
    }
    catch {
        Write-Host "[ERROR] $ScriptName threw an exception: $_" -ForegroundColor Red
        $exitCode = 1
    }

    $elapsed = (Get-Date) - $stepStart
    $elapsedStr = "{0:mm}m {0:ss}s" -f $elapsed

    if ($exitCode -eq 0) {
        Write-Host "[OK] $ScriptName completed in $elapsedStr" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "[FAILED] $ScriptName exited with code $exitCode after $elapsedStr" -ForegroundColor Red
        $continue = Read-Host "Continue with the rest of this sequence anyway? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            Write-Host "Stopping sequence." -ForegroundColor Red
            return $false
        }
        return $true
    }
}

# ----------------------------------------------------
# Sequences
# ----------------------------------------------------
function Run-LegacyCore {
    Write-Host "`n--- Legacy core (00-05): raw video -> cropped frames ---" -ForegroundColor Magenta
    $ok = $true
    $ok = $ok -and (Invoke-Step "00_CleanFolders.ps1")
    $ok = $ok -and (Invoke-Step "01_get_avi_hash.ps1")
    $ok = $ok -and (Invoke-Step "02_get_metadata_ffprob.ps1")
    $ok = $ok -and (Invoke-Step "03_get_avi_metadata_powershell.ps1")
    $ok = $ok -and (Invoke-Step "04_extract_frames_from_avi.ps1")
    $ok = $ok -and (Invoke-Step "05_extract_crop_frames_from_avi.ps1")
    return $ok
}

function Run-LegacyVisual {
    Write-Host "`n--- Legacy visual pipeline (06-10): classical cleanup + optional AI upscale ---" -ForegroundColor Magenta

    $ok = $true
    $ok = $ok -and (Invoke-Step "06_artifacts_removal.py")
    $ok = $ok -and (Invoke-Step "07_make_transparent_background.py")
    $ok = $ok -and (Invoke-Step "08_round_edges.py")
    if (-not $ok) { return $false }

    $smoothChoice = Read-Host "Run 10_smooth_gradients.py before Real-ESRGAN? (y/N - recommended if you want 09 to run on smoothed data, not the raw blocky frames)"
    if ($smoothChoice -eq "y" -or $smoothChoice -eq "Y") {
        $ok = $ok -and (Invoke-Step "10_smooth_gradients.py")
        if (-not $ok) { return $false }
        $ok = $ok -and (Invoke-Step "09_Final_improved_real-esrgan.py" @("--source-folder", "06_Smoothed_Gradients"))
    }
    else {
        $ok = $ok -and (Invoke-Step "09_Final_improved_real-esrgan.py")
    }
    return $ok
}

function Run-AimlDataBranch {
    Write-Host "`n--- AIML data branch (aiml_06 - aiml_08 split) ---" -ForegroundColor Magenta
    $ok = $true
    $ok = $ok -and (Invoke-Step "aiml_06_artifacts_removal_lama.py")
    $ok = $ok -and (Invoke-Step "aiml_07_generate_synthetic_backgrounds.py")
    $ok = $ok -and (Invoke-Step "aiml_08_prepare_training_split.py" @(
            "--input-images-folder", "05_Synthetic_Backgrounds/images",
            "--masks-folder", "05_Synthetic_Backgrounds/masks"
        ))
    return $ok
}

function Run-Training {
    Write-Host "`n--- Train segmentation model ---" -ForegroundColor Magenta
    $device = Read-Host "Train on (c)pu or (g)pu? [c/G]"
    if ($device -eq "g" -or $device -eq "G") {
        Write-Host "GPU training on this machine is not started from here - submit aiml_08_submit_gpu_training.sh"
        Write-Host "to your university's SLURM cluster instead (see that script's header for setup steps)." -ForegroundColor Yellow
        return $true
    }
    return (Invoke-Step "aiml_08_train_segmentation_demo.py" @(
            "--images-dir", "../Images/04_Dataset/train/images",
            "--masks-dir", "../Images/04_Dataset/train/masks",
            "--device", "cpu"
        ))
}

function Run-Inference {
    Write-Host "`n--- Run trained model ---" -ForegroundColor Magenta
    $target = Read-Host "Run on (v)alidation set (scored) or (n)ew unlabeled frames? [V/n]"
    if ($target -eq "n" -or $target -eq "N") {
        return (Invoke-Step "aiml_09_run_trained_model_transparent.py")
    }
    else {
        return (Invoke-Step "aiml_09_run_trained_model_transparent.py" @(
                "--input-folder", "Images/04_Dataset/val/images",
                "--ground-truth-masks-folder", "Images/04_Dataset/val/masks"
            ))
    }
}

# ----------------------------------------------------
# Menu
# ----------------------------------------------------
function Show-Menu {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor White
    Write-Host " Project 2 Pipeline - Run Menu" -ForegroundColor White
    Write-Host "================================================" -ForegroundColor White
    Write-Host " 1) Legacy core            (00-05: video -> cropped frames)"
    Write-Host " 2) Legacy visual pipeline (06-10: classical + optional AI upscale)"
    Write-Host " 3) AIML data branch       (aiml_06 - aiml_08 split)"
    Write-Host " 4) Train model            (aiml_08 train)"
    Write-Host " 5) Run inference          (aiml_09)"
    Write-Host " 6) Run EVERYTHING in order (1 -> 2 and/or 3 -> 4 -> 5)"
    Write-Host " 0) Exit"
    Write-Host "================================================" -ForegroundColor White
}

$keepGoing = $true
while ($keepGoing) {
    Show-Menu
    $choice = Read-Host "Choose an option"

    switch ($choice) {
        "1" { Run-LegacyCore | Out-Null }
        "2" { Run-LegacyVisual | Out-Null }
        "3" { Run-AimlDataBranch | Out-Null }
        "4" { Run-Training | Out-Null }
        "5" { Run-Inference | Out-Null }
        "6" {
            if (Run-LegacyCore) {
                $branch = Read-Host "Run (l)egacy visual, (a)iml branch, or (b)oth? [l/a/B]"
                if ($branch -eq "l" -or $branch -eq "L") {
                    Run-LegacyVisual | Out-Null
                }
                elseif ($branch -eq "a" -or $branch -eq "A") {
                    if (Run-AimlDataBranch) {
                        if (Run-Training) {
                            Run-Inference | Out-Null
                        }
                    }
                }
                else {
                    Run-LegacyVisual | Out-Null
                    if (Run-AimlDataBranch) {
                        if (Run-Training) {
                            Run-Inference | Out-Null
                        }
                    }
                }
            }
        }
        "0" { $keepGoing = $false }
        default { Write-Host "Not a valid option." -ForegroundColor Yellow }
    }
}

Write-Host "`nDone."

# ----------------------------------------------------
# Later deployment phase (not built yet - notes only):
#   Once this pipeline is stable, a friendlier front-end could wrap this
#   same menu logic - e.g. a simple web UI (Flask/Streamlit) or a compiled
#   desktop app - for less command-line-comfortable users (labmates,
#   advisor, reviewers). The functions above (Run-LegacyCore,
#   Run-LegacyVisual, Run-AimlDataBranch, Run-Training, Run-Inference) are
#   deliberately kept as separate, single-purpose blocks so that logic can
#   be reused/called from a future UI layer rather than rewritten.
# ----------------------------------------------------