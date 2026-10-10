# Model Card: TinyUNet (Streamlined MVP Edition)
*Prepared for the JCB Pilot Hall and University Teaching Workflow*

This model card documents the specifications, intended use, training setup, and validation protocols for the **TinyUNet** segmentation model. This document serves as an auditable compliance record for the model lifecycle.

---

## 1. Model Details
* **Model Name:** TinyUNet
* **Architecture:** Lightweight 2D U-Net Convolutional Encoder-Decoder [99, 178]
* **Version:** 2.0 (Streamlined MVP Edition) [175]
* **Date:** August 2026 [175]
* **Task Type:** Binary Semantic Segmentation (Foreground/Background Extraction) [178]
* **Language/Framework:** PyTorch [135, 178]

---

## 2. Intended Use & Domain of Applicability
* **Primary Intended Use:** Automated preprocessing of 2D electrical resistance tomography (ERT) sensor frames. The model isolates the active mixing data region and outputs transparent RGBA frames [178].
* **Downstream Integration:** The predicted transparency masks directly feed the client-side WebGL/Three.js 3D slice-stacking visualizer (`16_cylinder_reconstruction.html`) to generate 3D volumetric representations of mixing tanks in web browsers [186].
* **Out-of-Scope Use:**
  * Post-processed cosmetic layers (such as cubic-spline smoothed or edge-rounded frames) must never be used as raw source inputs for training or quantitative scientific analysis [113, 191].
  * Quantitative volumetric reconstruction metrics are not supported, as the 3D model is currently generated via physical visual slice-stacking rather than a learned volumetric reconstruction network [115, 186].

---

## 3. Training & Computational Environment
* **Platform Architecture:** Hybrid infrastructure-native execution [189, 190].
* **Compute Resources:** 
  * Heavy training is offloaded as a batch script (`aiml_08_submit_gpu_training.sh`) to the university’s existing High-Performance Computing (HPC) cluster [111, 189].
  * Standard local Topographer workstations are used for local inference, crop operations, and web-based 3D visualization [190].
* **Reproducibility Controls:**
  * **Input Data Hashing:** Cryptographic SHA256 hashing of the raw `active.avi` video file via `01_get_avi_hash.ps1` for complete chain-of-custody tracking [110, 191].
  * **Environment Isolation:** Dependencies are locked in the `.venv-tomo` virtual environment to prevent package version drift [14].
  * **Deterministic Seeds:** Deterministic random seeds are locked across Python, PyTorch, and NumPy to ensure reproducibility of dataset splits and synthetic background generation [11].

---

## 4. Dataset & Preprocessing
* **Primary Source:** Extracted 2D cross-sectional tomographic frames cropped to the region of interest (ROI) [110].
* **Data Augmentation:** Since experimental scans are sparse (8-layer physical scans), the training set is synthetically augmented using 10 distinct procedural background types (`aiml_08_generate_synthetic_backgrounds.py`) to prevent overfitting and ensure robust boundary detection under varying ambient laboratory conditions [11, 178].
* **Data Splits:** Split into reproducible train and validation sets via `aiml_08_prepare_training_split.py` [11, 110].

---

## 5. Metrics & Performance Evaluation
* **Loss Function:** 
  * Trained using **Dice Loss** to measure spatial overlap between predicted masks ($X$) and ground-truth masks ($Y$) [185]:
    $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |X \cap Y|}{|X| + |Y|}$$
* **Validation Metric:** Intersection-over-Union (IoU) scoring on held-out validation frames [178, 188].
* **Key Performance Indicator (KPI):** Targeting an Image Segmentation Dice coefficient of $>0.90$ for full production deployment [119].

---

## 6. Ethical, Safety, & Governance Controls
* **Human-in-the-Loop Validation:** Requires 100% of processed frame sequences to undergo visual validation via the interactive Image Variation Gallery (`gallery.html`) to check for boundary erosion or model hallucinations before deployment [181, 188].
* **Manual Override:** System includes a manual override step to ensure expert technician oversight of all automated processing decisions [173].
* **Data Privacy:** Operates entirely on anonymised, non-personal sensor-generated conductivity data, ensuring complete UK GDPR compliance [173, 195].
