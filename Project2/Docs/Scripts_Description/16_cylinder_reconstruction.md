# Script Description: 16_cylinder_reconstruction.html

## Overview

| Field | Value |
|---------|---------|
| **File Name** | 16_cylinder_reconstruction.html |
| **Language** | HTML, CSS, JavaScript |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Interactive 3D Visualisation / Cylinder Reconstruction User Interface |
| **Platform** | Modern Web Browser |
| **Execution Type** | Standalone HTML Application |
| **Supersedes** | 15_cylinder_reconstruction.html |

## Categories

- **3D Scientific Visualisation** – Displays reconstructed tank slices in a stacked cylindrical volume.
- **Interactive Data Exploration** – Allows users to inspect reconstructed data from different perspectives.
- **Engineering User Interface** – Provides visual controls for dataset loading, playback, and display configuration.
- **Volumetric Reconstruction Viewer** – Converts 2D slice images into a visual 3D representation.
- **Research Presentation Tool** – Intended for review, demonstration, and communication of reconstruction results.

---

## Purpose

This HTML application is a corrected revision of `15_cylinder_reconstruction.html`, providing the final visualisation interface for reconstructed tank data generated during:

**Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**.

The application enables users to:

- Display multiple tomographic slices as a stacked cylinder.
- View reconstructed mixing conditions in an interactive 3D environment.
- Rotate, zoom, and pan around the reconstructed volume.
- Load custom frame datasets directly from a local folder.
- Step through reconstructed frames manually.
- Play frame sequences as an animation, with an adjustable playback speed.
- Toggle visibility of the tank shell and image slices independently.
- Adjust slice transparency (opacity).
- Enable automatic rotation for presentation purposes.
- Visualise reconstructed tank geometry using a transparent cylindrical shell.

The tool serves as the primary front-end visualisation environment for reviewing processed tomography imagery, and is the current recommended version for use in place of the earlier `15_cylinder_reconstruction.html` build.

---

## Change Summary (v15 → v16)

This revision was produced to correct a slice-ordering discrepancy identified in the previous build and to add institutional branding to the interface.

| Change | Description |
|--------|-------------|
| **Slice order correction** | Slices 7 and 8 (the two lowest slices in the stack) have been swapped relative to `15_cylinder_reconstruction.html`, so that the rendered stack now correctly matches the original tomography acquisition/grid layout at the bottom of the tank. The page title and on-screen overlay text explicitly flag this correction (`"...Reconstruction. 7 <-> 8"`). |
| **Overlay text update** | The information overlay now explicitly states that "Slices 7 and 8 have been swapped," so reviewers are aware the ordering was corrected from the prior version. |
| **Branding graphics added** | Embedded logo/scale graphics (base64-encoded PNG images) have been added to the information overlay, e.g. an institutional logo and a scale-reference image, improving presentation quality for meetings and publications. |
| **Control panel unchanged in function** | All controls present in v15 (tank shell toggle, slice toggle, opacity slider, auto-rotate, folder loading, frame slider/navigation, play/speed) are retained unchanged in v16. |

No other functional or structural changes were made to the reconstruction, rendering, or frame-processing logic.

---

## Business / Process Purpose

The user interface provides an engineering and research-focused visualisation capability.

Its business and project objectives include:

- **Improved understanding of mixing behaviour** through volumetric presentation.
- **Visual validation of reconstruction outputs** generated earlier in the processing pipeline.
- **Communication of results** to project stakeholders.
- **Presentation support** for research meetings, demonstrations, and publications.
- **Interactive exploration** of image-processing and reconstruction results.
- **Reduction of interpretation effort** when comparing individual 2D slices.
- **Correction of a slice-ordering error** present in the prior version, ensuring the displayed geometry accurately reflects the true tomography acquisition sequence.

By converting individual reconstructed slices into a cylindrical stack, the software allows users to interpret the spatial relationships between measurement planes more effectively than viewing isolated images.

---

## User Interface Features

### Main Visualisation Window

The main canvas uses a real-time 3D rendering engine (Three.js) to display:

- Stacked image slices, correctly ordered top-to-bottom (including the corrected 7/8 order).
- Cylindrical tank shell.
- Slice spacing and orientation.
- Frame-specific reconstruction data.
- Interactive camera controls.

### Information Overlay

Displays:

- Project title, including institutional logo graphic.
- Reconstruction description, noting the tank height-to-diameter ratio and the 7/8 slice-order correction.
- User instructions (drag to rotate, scroll to zoom, right-drag to pan).
- Current frame name.
- Tank geometry / scale reference graphic.

### Control Panel

Provides user-configurable options including:

| Control | Function |
|----------|----------|
| Show Tank Shell | Enable or disable cylinder shell visibility |
| Show Slices | Enable or disable image slice visibility |
| Slice Opacity | Adjust transparency of reconstructed slices |
| Auto Rotate | Continuously rotate the model |
| Load Frames Folder | Import local image datasets |
| Frame Slider | Select reconstruction frame |
| Previous / Next | Navigate frame sequence |
| Play | Automatic frame playback |
| Speed Control | Adjust playback speed |

---

## Reconstruction Method

The application reconstructs a cylindrical representation from individual tomography slices.

The process includes:

1. Load reconstructed image frames (from the embedded demo dataset or a user-selected folder).
2. Identify circular tomography regions within each frame image (in-browser blob detection and cropping).
3. Arrange detected slices in the corrected vertical sequence.
4. Position slices along the tank height axis.
5. Render each slice as a textured plane.
6. Overlay slices within a transparent cylinder shell.
7. Update geometry dynamically during playback.

The default arrangement consists of:

- 8 horizontal slices.
- Slice 1 positioned at the top.
- Slice 8 positioned at the bottom.
- **Slices 7 and 8 swapped relative to v15**, so the bottom two positions in the stack now correctly correspond to the original tomography grid layout used during image acquisition and processing.

---

## 3D Interaction

Users can interact with the reconstruction using standard mouse controls:

| Action | Function |
|----------|----------|
| Left Drag | Rotate model |
| Mouse Wheel | Zoom |
| Right Drag | Pan |
| Auto Rotate | Continuous rotation |

These controls allow detailed visual inspection of reconstructed mixing behaviour and slice alignment.

---

## Technology Assessment

| Aspect | Assessment |
|----------|----------|
| **Complexity** | High. Combines image handling, user interface management, real-time rendering, volumetric reconstruction, and animation control. |
| **Dependencies** | Three.js r128 (loaded from CDN). |
| **Error Handling** | Moderate. Includes file loading validation and user feedback (e.g. status messages when no circular slices are detected). |
| **Portability** | High. Runs in any modern browser without installation. |
| **Performance** | Excellent for typical tomography datasets. |
| **Security Considerations** | Local file processing only. No credentials or authentication required. |
| **Maintainability** | High. Self-contained implementation with separated UI, styling, and rendering logic; change from v15 is a small, well-documented correction. |

---

## External Libraries

### Three.js

The application uses:

- **Three.js r128**

for:

- Scene rendering
- Camera management
- Lighting
- Texture mapping
- Geometry generation
- Animation updates

Three.js provides the real-time WebGL rendering foundation used throughout the reconstruction viewer.

---

## Logging and Reporting

Unlike the earlier processing scripts, this application does not generate processing logs.

Instead, operational feedback is provided through:

- On-screen status messages (e.g. folder-loading confirmation, frame-processing failures).
- Frame indicators.
- Dataset loading notifications.
- Playback status controls.

No local log files are created.

---

## Expected Usage Workflow

```text
Image Processing Pipeline
        │
        ▼
Final Reconstructed Frames
        │
        ▼
16_cylinder_reconstruction.html
        │
        ▼
Interactive 3D Visualisation
        │
        ▼
Review / Analysis
```

The application is intended as the final visualisation stage of the reconstruction workflow, replacing `15_cylinder_reconstruction.html` for this purpose.

---

## Expected Folder Structure

```text
Project2\
│
├── Images\
│   ├── 04_Transparent\
│   ├── 05_Final_Rounded\
│   ├── 06_Final_Upscaled\
│   ├── 06_Inference_Output\
│   └── 06_Smoothed_Gradients\
│
├── Reports\
│
├── Logs\
│
└── Script\
    └── 16_cylinder_reconstruction.html
```

Users may load frame folders dynamically through the interface rather than requiring hardcoded paths.

---

## Approval Considerations

- No modification is performed on source images.
- No network communication occurs other than loading the Three.js library.
- Source frames remain unchanged.
- The application processes data in memory only.
- Suitable for academic and research environments.
- Provides a visual interpretation of reconstructed data rather than generating new measurement data.
- Intended primarily for visualisation and communication purposes.
- Corrects a slice-ordering discrepancy present in `15_cylinder_reconstruction.html`; the earlier build should be considered superseded.

---

## Suggested Classification

### Classification: Low Risk / Interactive Scientific Visualisation Tool

**Risk Level:** Low

**Primary Risks:**

- Misinterpretation of visual appearance as quantitative analysis.
- Incorrect assumption that visual interpolation represents additional measured data.
- Risk of using the superseded v15 build, which contains the uncorrected slice ordering.

**Change Category:**

- Standard engineering visualisation utility; minor corrective revision (slice-order fix and branding update).

**Data Sensitivity:**

- Derived scientific tomography imagery associated with mixing-tank reconstruction experiments.

**Recommended Handling:**

- Use `16_cylinder_reconstruction.html` in place of `15_cylinder_reconstruction.html` for all future review, presentations, demonstrations, project meetings, and publications.
- Retain source reconstruction data as the authoritative scientific record.
- Clearly identify the visualisation as a graphical reconstruction of processed tomography data.

---

## AI / Machine Learning Assessment

This user interface does **not** directly perform AI or Machine Learning operations.

The HTML application visualises outputs generated by previous processing stages, including:

- AI-enhanced images (Real-ESRGAN).
- Processed tomography slices.
- Reconstructed volumetric datasets.

For AI/ML Fellowship evidence purposes, this file should be classified as:

**Visualisation and Human-Machine Interface (HMI) Layer**

rather than an AI/ML implementation component.
