# TomoDockerPipeline UI

## Installation
Extract `TomoDockerPipeline` into the Project3 root beside `Scripts`, `Models`, `Utilites`, `Images` and `Video`.

Expected model filenames:
- `Models/TinyUnet_Model.pt`
- `Models/big-lama.pt`
- `Models/RealESRGAN_x4plus.pth`

If your folder is named `Utils`, correct the `COPY Utils/...` line in the Dockerfile.

## Start
From the `TomoDockerPipeline` folder run `docker compose up --build`, then open `http://localhost:18080`.

## Data
Each run is isolated under `Experiments/<experiment name>/`. The only accepted input is one AVI video. Generated outputs are PNG images plus logs and reports. TinyUNet runs on every pipeline run. LaMa and Real-ESRGAN are optional.
