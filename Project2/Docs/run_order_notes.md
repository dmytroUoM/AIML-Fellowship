# Run Order Notes — Project 2 Scripts

Quick reference only. See each script's own documentation / `--help` for full parameter details.

---

## 1. Legacy pipeline (core, always runs first)

Everything else depends on this running first — it turns the raw video into cropped frames.

```
00_CleanFolders.ps1
01_get_avi_hash.ps1
02_get_metadata_ffprob.ps1
03_get_avi_metadata_powershell.ps1
04_extract_frames_from_avi.ps1
05_extract_crop_frames_from_avi.ps1
```

→ Output: `Images\02_Frames`

---

## 2. Legacy visual pipeline (classical, no AI)

Continues from `05`. Produces presentation-ready frames using classical + Real-ESRGAN processing.

```
06_artifacts_removal.py            (magenta-threshold cleanup)
07_make_transparent_background.py
08_round_edges.py
09_Final_improved_real-esrgan.py   (AI super-resolution — optional, presentation only)
10_smooth_gradients.py             (optional — run on 05_Final_Rounded OR post-esrgan, your call; see note below)
```

**Note on `10_smooth_gradients.py`:** it's optional and decoupled from strict order on purpose (per your call — no major scientific benefit smoothing native-resolution blocky data). If you *do* want smoother Real-ESRGAN output, run it **before** `09`, not after — feeding blocky data into Real-ESRGAN causes it to sharpen the blocks rather than fix them.

→ Final output: `Images\06_Final_Upscaled` (or wherever you pointed `09`'s `--output-folder`)

---

## 3. AIML branch (parallel alternative to 06-10 above, starts from the same `05` output)

```
aiml_06_artifacts_removal_lama.py           (LaMa-based cleanup, jagged-mask-aware)
aiml_07_generate_synthetic_backgrounds.py   (varied backgrounds for training data)
aiml_08_prepare_training_split.py           (train/val split)
```

→ Output: `Images\04_Dataset\train\...` and `...\val\...`

---

## 4. Train + run the segmentation model

```
aiml_08_train_segmentation_demo.py          (CPU or GPU — see --device)
aiml_08_submit_gpu_training.sh              (same step, SLURM job for university GPU cluster)
```
→ Output: `results/<run-name>_model.pt`, `results/<run-name>_history.json`

```
aiml_09_run_trained_model_transparent.py    (inference — run on val set first for real IoU score, THEN on new frames)
```

---

## One-line summary

```
05 (legacy, required)
 ├─ 06 → 07 → 08 → [10] → 09      (legacy visual branch, classical + optional AI upscale)
 └─ aiml_06 → aiml_07 → aiml_08 (split) → aiml_08 (train) → aiml_09   (AI/ML training branch)
```
