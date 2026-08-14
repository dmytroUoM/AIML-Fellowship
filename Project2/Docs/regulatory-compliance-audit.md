# Regulatory Compliance, Risk Management, and Model Lifecycle Audit
**Document Version:** 1.0 (Audit & Lifecycle Edition)  
**Date:** August 2026  
**Author:** Senior Technician (Control Systems) / AI ML Fellowship Apprentice  
**Status:** Approved for Academic and Operational Integration  

---

## 1. Internal Compliance Audit Report [S17]

An internal compliance audit was conducted on the Project 2 Machine Learning (ML) pipeline—**"AI-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks"**—to verify strict adherence to institutional legal, ethical, and corporate governance requirements. The findings indicate high compliance with established standards, along with specific action items to advance operational maturity.

### A. Compliance Matrix & Verification

| Regulation / Standard | Compliance Domain | Evaluation & Working Evidence | Audit Status |
| :--- | :--- | :--- | :--- |
| **UK GDPR & Data Protection Act 2018** | Data Privacy & Protection | The pipeline processes exclusively anonymised, non-personal sensor-generated electrical conductivity data. No personally identifiable information (PII) is captured, stored, or processed, mitigating privacy-related legal liabilities by design. | **Compliant** |
| **ISO 9001:2015** | Quality Management Systems | Comprehensive quality standards are maintained through structured, sequentially numbered PowerShell and Python processing scripts (Step 0 to Step 11). Every pipeline stage outputs timestamped execution logs with standard severity levels (`INFO`, `SUCCESS`, `WARNING`, `ERROR`) to a dedicated `Logs` directory. | **Compliant** |
| **ISO/IEC 27001 / 27002** | Information Security | High security is achieved through local-only, air-gapped execution on Topographer workstations in the JCB Pilot Hall. Heavy segmentation model training is offloaded to the university’s managed High-Performance Computing (HPC) cluster via secure SLURM scripts (`aiml_08_submit_gpu_training.sh`). | **Compliant** |
| **Academic Ethics & Scientific Integrity** | Responsible Research & Transparency | Scientific objectivity is enforced by segregating cosmetic enhancements (e.g., cubic-spline smoothed gradients, edge rounding) into separate, explicitly labelled directories (`05_Final_Rounded`, `06_Smoothed_Gradients`). This prevents the training of models on cosmetically altered "hallucinated" data. | **Compliant** |

### B. Identified Areas for Improvement & Action Plan

Despite high fundamental compliance, the audit identified three key operational risks:
1. **Manual Backup Redundancy:** Operational data and trained weights are backed up manually on an annual hard drive (HDD) cloning schedule, exposing the system to a high risk of catastrophic data loss between cloning cycles.
   * *Correction:* Implement an automated, script-based weekly backup utility that copies the `Reports`, `Logs`, and `Images` directories to a secure local external storage drive.
2. **Hardcoded Configuration Paths:** The core pipeline scripts target a fixed, hardcoded input filename (`active.avi`) and folder structure. This limits automated throughput and batch processing.
   * *Correction:* Refactor scripts to accept dynamic command-line interface (CLI) parameters, facilitating batch orchestration to scale capacity to the target 36 samples per day.
3. **Metric Standardisation Gap:** The custom `TinyUNet` segmentation model reports performance using Intersection-over-Union (IoU), whereas the primary success metric of the original project proposal is expressed as a Dice coefficient (>0.90).
   * *Correction:* Add a Dice metric computation function to `aiml_09_run_trained_model_transparent.py` to allow direct, un-interpolated comparison against the stated KPIs.

---

## 2. Comprehensive ML Model Lifecycle Document [S6]

This section traces the end-to-end lifecycle of the `TinyUNet` segmentation model and its associated image-processing pipeline, from initial development to laboratory deployment.

```
[Development: HPC SLURM Training] ──> [Data Prep: Hashing & ROI Crop] ──> [ML Branch: TinyUNet Inference]
                                                                                      │
[Deployment: Pilot Hall Lab (Air-Gapped)] <── [Monitoring: gallery.html Audit] <──────┘
```

### Phase A: Development & Model Engineering
* **Algorithm Selection & Parameter Capacity:** A compact `TinyUNet` architecture (~117K parameters) was selected as a pragmatic, highly efficient 2D segmentation encoder-decoder. This lightweight design avoids the memory overloads and application crashes associated with legacy 3D EIT software packages, offering a 10× reduction in inference latency (30 seconds vs 300 seconds for a full 3D U-Net) with an acceptable 6.4% trade-off in spatial accuracy.
* **Deterministic Training and Reproducibility:** To guarantee scientific reproducibility, the training process locks deterministic random seeds across Python, PyTorch, and NumPy. Dependencies and exact package versions are strictly encapsulated within a dedicated virtual environment (`.venv-tomo`).
* **Synthetic Data Augmentation:** To prevent model overfitting on sparse physical scans (only 8 physical layers are captured by legacy hardware), the raw cropped frames are composited onto 10 distinct procedural, synthetic background types (`aiml_08_generate_synthetic_backgrounds.py`) to build a robust augmented training set.
* **Distributed Compute Security:** Training is offloaded as a secure batch job (`aiml_08_submit_gpu_training.sh`) to the university’s shared HPC cluster, leveraging institutional SLURM access controls and eliminating local hardware capital expenditure.

### Phase B: Preprocessing & Data Extraction
* **Cryptographic Data Provenance:** At Step 1, a PowerShell script (`01_get_avi_hash.ps1`) computes and archives a SHA256 cryptographic hash of the raw `active.avi` video file. This creates an unalterable chain of custody, enabling automated detection of data corruption or unauthorised manipulation before downstream processing.
* **Metadata Extraction & Frame Decoding:** Detailed stream metadata is parsed using `ffprobe.exe` (saved to `02_avi_metadata_ffprobe.json`) and Windows Explorer Shell COM objects. Raw frames are decoded into lossless PNGs inside the version-controlled `Images/01_Origin` directory.
* **Region of Interest (ROI) Cropping:** Extracted frames are dynamically cropped to the active mixing vessel boundaries, producing cleaned subsets in `Images/02_Frames`.

### Phase C: Inference & Post-Processing
* **Hybrid Core Models:**
  1. *LaMa Deep-Learning Inpainting:* Automatically detects and removes colored text and instrumentation numbering overlays outside the circular data plots using fast Fourier convolutions (FFCs), writing outputs to `03_Cleaned_lama`.
  2. *Real-ESRGAN Segmenter:* Generates highly precise binary masks of the active mixing region to output transparent RGBA frames in `Images/06_Inference_Output/images`.
  3. *Alpha-Aware Real-ESRGAN:* Upscales and sharpens the low-resolution frames while processing the transparency (alpha) and RGB channels in parallel, separate streams to prevent edge-fringe artefacts.
* **Cosmetic Post-Processing Segregation:** A presentation-only layer applies cubic-spline resampling to smooth blocky EIT colour transitions. These cosmetically modified files are restricted to folder `06_Smoothed_Gradients` and are formally documented as "not-for-reuse" in quantitative scientific evaluation or retraining to prevent dataset contamination.

### Phase D: Deployment & Operations (Pilot Hall Constraints)
The pipeline is deployed in the second-year Chemical Engineering teaching laboratory under highly constrained operational parameters:
* **Physical Access Controls:** The tomography workstation is physically isolated within a dedicated room in Pilot Hall, accessible exclusively via university security-issued access cards.
* **Network Air-Gapping:** The workstation operates in a complete air-gapped state with no active internet access, eliminating remote cyber threats or network-based data exfiltration.
* **Logical Access Controls:** The workstation is password-protected and implements separate "User" and "Admin" accounts. This establishes role-based access control (RBAC), in which standard operators run model inference and perform visual audits under user-level privileges, and administrative elevation is required to modify pipeline scripts.
* **Engineering Endpoint Protection:** Model development and engineering scripts are maintained on university-managed laptops, ensuring device compliance and routine administrative security patching.
* **Resilience & Backup Protocols:** Currently, data backups are performed manually with hard drive (HDD) cloning executed once a year. The post-deployment operational plan defines a transition to an automated daily/weekly backup script to secure experimental records.

---

## 3. Justification of Documentation and Tracking Methods [S32]

A strong cybersecurity and operational culture relies on transparent, verifiable documentation. The chosen documentation methods for Project 2 demonstrate a commitment to rigorous machine learning engineering:

1. **The Model Card Standard (`tinyunet-model-card.md`):** Adopting the industry-standard Model Card framework ensures that critical model parameters, intended domains of applicability, performance trade-offs, and dataset characteristics are transparently exposed to stakeholders and external regulators. This prevents the deployment of "black-box" models and establishes a clear baseline for model compliance.
2. **Directory-Level Version Isolation:** Implementing numbered, sequentially dependent output subdirectories (`01_Origin`, `02_Frames`, `03_Cleaned_lama`, etc.) serves as a physical manifest of data transformation. It provides an immediate visual and programmatic method for auditing how data flows through the pipeline, simplifying tracking of feature engineering.
3. **Structured Pipeline Logs:** Writing timestamped, levelled logs (`INFO`, `SUCCESS`, `WARNING`, `ERROR`) to a centralised `Logs` folder ensures non-repudiation. In the event of system anomalies or pipeline failures, technicians can easily identify the exact processing stage that failed, thereby fulfilling the operational quality requirements of ISO 9001:2015.

---

## 4. Explicit Adherence to ML Principles and Organisational Policies [S32]

The Project 2 solution is explicitly mapped to responsible innovation and ethical AI principles, leveraging both the **AREA** and **SAFE-D** governance frameworks to translate abstract ethical guidelines into concrete engineering practices.

### A. The AREA Framework for Responsible Innovation

* **Anticipate:** Potential technical failure modes—such as model-induced boundary erosion, artificial hallucinations, and the loss of scientific data (e.g., active mixing dead zones) due to automated inpainting—were anticipated during the design phase. This led to the creation of the **Interactive Image Variation Gallery (`gallery.html`)**.
* **Reflect:** The team reflected on the scientific integrity of "prettified" data. If cosmetic enhancements (such as cubic-spline smoothing or edge rounding) were mistaken for physical measurements, this could lead to false quantitative conclusions in materials science research. This reflection prompted a strict policy of folder isolation and mandatory "visualisation-only" labelling for derivative folders.
* **Engage:** Broad stakeholder engagement was central to the pilot. The team actively engaged with Graduate Teaching Assistants (GTAs), academic staff, and Pilot Hall Team to assess how the transition from legacy 2D pixelated images to 3D volumetric stacks would improve student comprehension, reduce interpretation bottlenecks, and support neurodiverse students.
* **Act:** Concrete actions were taken based on engagement findings. The legacy slice-ordering error (slices 7 and 8 ordered incorrectly) was resolved in the final production release of `16_cylinder_reconstruction.html`, and a **manual override capability** was built into the interface to ensure that expert operators can bypass automated segmentation errors.

### B. The SAFE-D Governance Framework

* **Safety:** Achieved by operating local-only, sandboxed execution on air-gapped laboratory workstations, preventing external network attacks and securing proprietary research data.
* **Accountability:** Built into the pipeline via cryptographic SHA256 hashing of the raw input AVI data and structured, non-repudiable logs that trace every execution step.
* **Fairness:** The development of the WebGL/Three.js 3D cylindrical reconstruction tool makes abstract, noisy tomographic data highly accessible, reducing cognitive load and lowering barriers for neurodiverse students who struggle to reconstruct 3D spatial mixing patterns from flat 2D slices.
* **Transparency, Traceability and Data Integrity:** The preprocessing pipeline generates intermediate binary masks (03_Masks_lama) that allow verification of the regions selected for artefact removal. Only non-informative elements, such as text overlays, borders, and background colours, are removed, while the original EIT measurements and tomographic reconstructions remain unchanged. Data provenance, hashing, and validation checks provide full traceability, reproducibility, and confidence in the integrity of the processed dataset.
* **Data Governance:** Enforced through systematic, versioned file directories, preventing loss of traceability and establishing unambiguous provenance for every processed image.

---
*End of Audit Report.**
