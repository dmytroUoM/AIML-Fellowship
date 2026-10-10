# Project 3: Minimal AI Inference Pipeline

Project 3 is a minimal, inference-only image-processing pipeline for mixing-tank tomography. It takes an AVI video, extracts PNG frames, and runs three pretrained AI stages:

1. **TinyUNet** for foreground segmentation
2. **LaMa** for image inpainting and artefact removal
3. **Real-ESRGAN** for image enhancement and super-resolution

There is no model-training workflow in Project 3. The same Python scripts can run directly in **PowerShell** or inside **Docker** on a Windows machine.

## Minimal project structure

```text
Project3/
|-- Experiments/
|   `-- Group1Exp01/
|       |-- Images/
|       |-- Reports/
|       `-- Video/
|
|-- Images/
|   |-- 01_Origin/
|   |-- 02_Frames/
|   |-- 03_TinyUnet_Output/
|   |-- 04_Cleaned/
|   |-- 04_Masks/
|   |-- 05_Transparent/
|   |-- 06_LaMa_Inference_Output/
|   `-- 07_Real-ESRGAN_Inference_Output/
|
|-- Logs/
|-- Models/
|   |-- TinyUnet_Model.pt
|   |-- big-lama.pt
|   `-- RealESRGAN_x4plus.pth
|
|-- Reports/
|-- Scripts/
|   |-- 01_CleanFolders.py
|   |-- 02_Get_Avi_SHA256_Hash.py
|   |-- 03_Get_Video_Metadata.py
|   |-- 04_Get_Video_Frames.py
|   |-- 05_Run_TinyUNet_Inference.py
|   |-- 06_Run_LaMa_Inference.py
|   |-- 07_Real-ESRGAN_Inference_Output.py
|   |-- requirements.lock.txt
|   `-- run_pipeline.py
|
|-- TomoDockerPipeline/
|   |-- templates/
|   |-- app.py
|   |-- docker-compose.yml
|   |-- Dockerfile
|   `-- requirements-ui.txt
|
|-- Utils/
|   `-- requirements.txt
|
|-- Video/
|   `-- active.avi
|
`-- README.md
```

The root folders are the standard workspace. `Experiments/` provides an optional structure for keeping separate group or experiment outputs.

## Pipeline flow

```text
Video/active.avi
    |
    v
Images/01_Origin
    |
    v
Images/02_Frames
    |------------------|
    v                  v
TinyUNet             LaMa
    |                  |
    v                  v
03_TinyUnet_Output   06_LaMa_Inference_Output
                       |
                       v
                   Real-ESRGAN
                       |
                       v
                   07_Real-ESRGAN_Inference_Output
```

All execution logs are written to `Project3/Logs/`, and reports are written to `Project3/Reports/`. The detailed folder layout is based on the current Project 3 structure. 

# Deploy on a Windows machine

You can use either the local Python setup or Docker. Python is convenient for development and individual scripts. Docker is the simplest option for a consistent deployment on another Windows computer.

## 1. Copy or clone Project 3

Using Git:

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd .\AI_ML_Fellowship\Project3
```

Alternatively, copy the complete `Project3` folder to the target Windows machine.

Before running the pipeline, add:

```text
Video/active.avi
Models/TinyUnet_Model.pt
Models/big-lama.pt
Models/RealESRGAN_x4plus.pth
```

The model files are not included in Git and must be copied separately.

# Option A: Run with Python and PowerShell

## Requirements

- Windows 10 or Windows 11
- Python 3.11
- PowerShell

A Python virtual environment keeps this project's packages isolated from the system Python and other projects. [Python's venv documentation](https://docs.python.org/3/library/venv.html) describes this isolation model.

## Create the environment

From `Project3`:

```powershell
py -3.11 -m venv .venv-project3
.\.venv-project3\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv-project3\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\Utils\requirements.txt
```

Verify the environment:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

The interpreter should point to:

```text
Project3\.venv-project3\Scripts\python.exe
```

## Run the menu

```powershell
cd .\Scripts
python .\run_pipeline.py
```

> Use `python`, not `py`, after activating the environment. The `py` launcher can select the system Python instead of `.venv-project3`.

The menu provides:

```text
1. Preprocess video and create frames
2. Run TinyUNet
3. Run LaMa
4. Run Real-ESRGAN
5. Run the complete pipeline
```

## Run one script directly

From the `Project3` root:

```powershell
python .\Scripts\05_Run_TinyUNet_Inference.py
python .\Scripts\06_Run_LaMa_Inference.py
python .\Scripts\07_Real-ESRGAN_Inference_Output.py
```

# Option B: Run with Docker Desktop

Docker packages the application and its Python dependencies into a reusable container. Docker Desktop supports a WSL 2 backend on Windows, and Docker documents WSL 2 as the default backend for most Windows users. [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) provides the current requirements, while [Docker's WSL 2 guide](https://docs.docker.com/desktop/features/wsl/) explains the backend configuration.

## Requirements

- Windows 10 or Windows 11
- WSL 2
- Docker Desktop using Linux containers

Verify the installation:

```powershell
wsl --version
docker version
docker compose version
```

## Build and start Project 3

From the Project 3 root:

```powershell
cd .\TomoDockerPipeline
docker compose build --no-cache
docker compose up -d
```

Check the container:

```powershell
docker compose ps
docker compose logs -f
```

Open the interface at:

```text
http://localhost:18080
```

## Stop Project 3

```powershell
docker compose down
```

## Rebuild after changing scripts or configuration

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d
```

# Move Project 3 to another Windows machine

## Recommended method

1. Install WSL 2 and Docker Desktop on the destination computer.
2. Clone or copy the `Project3` folder.
3. Copy the three model files into `Project3/Models/`.
4. Place the input video at `Project3/Video/active.avi`.
5. Open PowerShell in `Project3/TomoDockerPipeline`.
6. Build and start the container:

```powershell
docker compose build --no-cache
docker compose up -d
```

7. Open `http://localhost:18080`.

This rebuilds the Docker image on the destination machine and avoids transferring machine-specific virtual environments or cached container data.

# Quick checks

Check the required input and models from the Project 3 root:

```powershell
Test-Path .\Video\active.avi
Test-Path .\Models\TinyUnet_Model.pt
Test-Path .\Models\big-lama.pt
Test-Path .\Models\RealESRGAN_x4plus.pth
```

Each required file should return `True`.

Check whether port `18080` is already occupied:

```powershell
netstat -ano | findstr :18080
```

If necessary, change the host-side port in `docker-compose.yml`, for example from `18080:18080` to `18081:18080`, and then open `http://localhost:18081`.

# In short

- Use **PowerShell** when you want direct access to individual Python scripts.
- Use **Docker** when you want the easiest repeatable deployment on another Windows machine.
- Keep input data in `Video/`, models in `Models/`, outputs in `Images/`, reports in `Reports/`, and execution history in `Logs/`.
