# Project Charter: AI-Powered 3D EIT Mixing Tank Reconstruction

## 1. Project Overview
This project overcomes the limitations of traditional Electrical Impedance Tomography (EIT) software by implementing an AI/ML-powered workflow. It converts conventional 2D tomographic slices of horizontal plates into interactive, high-resolution 3D volume reconstructions, all accessible through a browser-based interface.

### 1.1 Problem Statement
* **Lack of 3D Visualization:** The legacy EIT software produces only isolated 2D cross-sectional images, which restricts comprehensive spatial analysis of mixing processes.
* **Performance & Memory Constraints:** The native high-resolution imaging options cause severe memory overload, leading to frequent application crashes.
* **Low Engagement:** Processing pixelated 2D images takes significant time and reduces user engagement among researchers and students.
* **Issue in Slices 7 and 8:** There is a known discrepancy where slice 7 in the legacy tomogram actually corresponds to slice 8, causing potential inaccuracies and complicating data interpretation.

### 1.2 Core Objective
To build a lightweight, stable, and browser-based 3D reconstruction application using modern computer vision and machine learning techniques, achieving high-resolution visualizations **without requiring software or hardware infrastructure upgrades**.

---

## 2. Project Scope & Deliverables
* **3D Volume Reconstruction:** Recreating the mixing tank environment by stacking and interpolating 2D horizontal plate datasets.
* **AI-Driven Super-Resolution:** Using deep learning to upscale pixelated 2D images smoothly, preventing memory crashes during high-resolution tasks.
* **Interactive Web Interface:** A modern browser application allowing users to rotate, zoom, and analyze the full 3D tank volume in real-time.

---

## 3. Technology Stack & Tools
The development workflow leverages modern software engineering, data science, and automation tools:

* **Programming Language:** Python (with core scientific libraries)
* **Computer Vision:** OpenCV (image processing and transformations)
* **AI Upscaling:** Real-ESRGAN (high-resolution image enhancement)
* **Media Handling:** FFmpeg (video rendering and frame processing)
* **Methodology:** LLM-assisted development workflows
* **Automation:** PowerShell scripts for data pipeline automation
* **Version Control & Management:** GitHub

---

## 4. Success Criteria & Business Value

### 4.1 Technical & Operational Impact
* **Infrastructure Savings:** Zero cost on hardware upgrades by shifting heavy processing loads to optimized AI models.
* **System Stability:** Elimination of memory-related application crashes during high-resolution processing.
* **Efficiency:** Faster overall processing times compared to legacy 2D analysis workflows.

### 4.2 Educational & Accessibility Impact (Neurodiversity)
* **Enhanced Service Delivery:** A massive leap forward in usability, transforming data into intuitive visual formats.
* **Inclusive Learning:** Improved data accessibility for students and researchers who process 2D and 3D visual/spatial information differently (supporting neurodiverse learning styles).

---

## 5. Deployment & Review Timeline
1. **Prototype Phase (Current):** Internal deployment of the working core model.
2. **First Review Session:** Stakeholder evaluation focused on usability, interface refinement, and accessibility impact.
3. **Refinement:** Implementation of stakeholder feedback.
4. **Final Review Session:** Validation of stability and readiness for routine academic and research use.
5. **Full Deployment:** Official integration into the research workflow.