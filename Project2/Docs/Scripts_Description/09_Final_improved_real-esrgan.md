## Script Description: 09_Final_improved_real-esrgan.py

### Overview

| Field | Value |
|---------|---------|
| **Script Name** | 09_Final_improved_real-esrgan.py |
| **Language** | Python |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | AI Image Enhancement / Super-Resolution / Scientific Visualisation |
| **Platform** | Cross-platform (Python) |
| **Execution Type** | Standalone Python script, manually executed or scheduled, supports extensive command-line parameters |

### Categories

- **AI-Powered Super-Resolution** — Uses the pretrained Real-ESRGAN RRDBNet model to enhance and upscale scientific image frames.
- **Scientific Image Visualisation** — Produces visually improved images for presentation, publication, and review purposes.
- **Batch Processing Automation** — Processes all PNG frames within a specified source folder without user interaction.
- **Transparency-Aware Processing** — Separately processes alpha channels to preserve clean transparent backgrounds.
- **Logging & Traceability** — Generates structured logs including processing statistics, execution details, and runtime measurements.

### Script Purpose

This script performs the final AI-based image enhancement stage within the Project 2 image-processing pipeline.

The script:

- Resolves the project structure automatically using relative paths.
- Loads a pretrained Real-ESRGAN model file.
- Detects and processes PNG images from the specified source folder.
- Applies AI-driven super-resolution to RGB image content.
- Processes transparency information independently from the AI model.
- Resizes alpha channels using configurable interpolation methods.
- Recombines enhanced RGB content with the resized alpha channel.
- Saves the resulting images into a configurable output folder.
- Logs processing statistics, configuration settings, and execution duration.
- Supports CPU and GPU execution.
- Supports tiled processing to reduce memory consumption.
- Exposes AI model parameters through command-line arguments for experimentation and optimisation.

### Business / Process Purpose

This script provides the final visual enhancement stage for:

**Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**

Its primary business and research objectives include:

- **Improved visual quality**: Enhances image sharpness, edge clarity, and presentation quality for reports, demonstrations, and publications.
- **AIML integration**: Demonstrates practical application of pretrained deep-learning models within an engineering workflow.
- **Scalable processing**: Enables automated enhancement of entire image datasets without manual editing.
- **Research reproducibility**: Records configuration settings and processing details through structured logging.
- **Controlled transparency handling**: Ensures transparent backgrounds remain clean and usable in downstream visualisation workflows.
- **Parameter experimentation**: Supports investigation of AI enhancement settings as evidence of machine-learning deployment and optimisation activities.

### AI / Machine Learning Use

This script represents a direct application of Artificial Intelligence and Machine Learning technologies within the project.

The AI functionality includes:

- Use of the pretrained **Real-ESRGAN** deep-learning model.
- Use of the **RRDBNet** neural-network architecture.
- Deep-learning-based image enhancement and super-resolution.
- Learned reconstruction of high-frequency visual details.
- GPU acceleration when CUDA-compatible hardware is available.
- Configurable inference settings including tiling, scaling, and precision modes.

The script does **not** perform:

- Model training
- Fine-tuning
- Transfer learning
- Dataset labelling
- Model evaluation

Instead, it performs **AI inference**, applying an already-trained model to new image data.

### Transparency Handling

Transparent image regions require special treatment during AI processing.

To prevent image artefacts:

- RGB channels are processed through Real-ESRGAN.
- Alpha channels bypass the AI model entirely.
- Alpha channels are resized using standard image interpolation.
- The enhanced RGB data and resized alpha data are recombined into a final RGBA image.

This approach prevents:

- Transparency haze
- Edge contamination
- Artificial texture generation in transparent areas
- Unwanted halo effects around image regions

### Configurable Parameters

The script exposes multiple command-line options:

| Parameter | Purpose |
|------------|------------|
| `--source-folder` | Input image folder |
| `--output-folder` | Output image folder |
| `--device-mode` | CPU, CUDA, or automatic device selection |
| `--model-name` | Real-ESRGAN model identifier |
| `--model-path` | Explicit path to model file |
| `--model-scale` | Native model scale factor |
| `--outscale` | Final output scaling factor |
| `--tile-cpu` | CPU tile size |
| `--tile-gpu` | GPU tile size |
| `--tile-pad` | Tile overlap padding |
| `--pre-pad` | Image preprocessing padding |
| `--use-half-on-gpu` | Enable FP16 inference |
| `--no-half-on-gpu` | Disable FP16 inference |
| `--alpha-interpolation` | Transparency interpolation method |
| `--no-log` | Disable file logging |

### Expected Folder Structure

```text
Project2\
│
├── Images\
│   ├── 05_Final_Rounded\
│   └── 06_Final_Upscaled\
│
└── Scripts\
    ├── 09_Final_improved_real-esrgan.py
    ├── RealESRGAN_x4plus.pth
    └── Logs\
```

### Logging the Result

The script maintains traceability by recording execution activity.

- **Log file location**: `<ScriptDir>\Logs\09_Final_improved_real-esrgan.log`
- **Log format**:

```text
[yyyy-MM-dd HH:mm:ss] [LEVEL] Message
```

- **Log levels used**:
  - INFO
  - WARNING
  - ERROR
  - SUCCESS

- **Information captured**:
  - Script start and completion
  - Input and output locations
  - Device selection
  - Model configuration
  - Processing progress
  - Per-image status
  - Processing failures
  - Total execution time
  - Runtime summary statistics

### Technology Assessment

| Aspect | Assessment |
|----------|----------|
| **Complexity** | High. Combines deep-learning inference, image processing, transparency management, argument parsing, logging, and hardware acceleration. |
| **Dependencies** | PyTorch, Real-ESRGAN, BasicSR, GFPGAN, FaceXLib, Pillow, NumPy, OpenCV. |
| **Error Handling** | Moderate to High. Handles dependency failures, configuration issues, image-processing exceptions, and logging errors. |
| **Portability** | Cross-platform Python implementation with optional GPU acceleration. |
| **Idempotency** | Yes. The script can be re-run multiple times with consistent results using the same model and parameters. |
| **Security Considerations** | Local processing only. No network communication or credential handling. |
| **Performance** | Moderate to High depending on image size and hardware. GPU acceleration significantly improves processing speed. |
| **Maintainability** | High. Parameterised design, structured logging, clear separation of image and transparency processing. |

### Approval Considerations

- AI-generated detail may not represent genuine measured features.
- Super-resolution output should not automatically replace original scientific data.
- Original cleaned images should remain the reference dataset.
- GPU execution requires local availability of CUDA-compatible hardware and libraries.
- Model files should be controlled and version tracked.
- Output images should be clearly identified as AI-enhanced derivatives.
- Processing occurs entirely on local systems with no external communications.
- Source images are read-only and remain unchanged.

### Suggested Classification

**Classification: Medium Risk / AI-Enhanced Scientific Visualisation Utility**

- **Risk Level:** Medium
  - No cybersecurity concerns.
  - No external communication.
  - No modification of source data.

- **Primary Risk:**
  - AI-generated visual details may be interpreted as genuine scientific information if adequate controls are not maintained.

- **Change Category:**
  - Standard engineering software utility with documented AI processing.

- **Data Sensitivity:**
  - Derived from scientific imaging data related to mixing-tank reconstruction experiments.

- **Recommended Handling:**
  - Use primarily for visualisation, reporting, presentation, and communication purposes.
  - Retain original cleaned images as the authoritative source of scientific truth.
  - Clearly label outputs as AI-enhanced imagery whenever used in publications, presentations, or project reports.