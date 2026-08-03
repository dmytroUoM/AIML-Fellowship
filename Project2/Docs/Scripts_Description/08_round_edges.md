# Script Description: 08_round_edges.py

## Overview

| Field | Value |
|---|---|
| **Script Name** | 08_round_edges.py |
| **Language** | Python |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | Cosmetic Image Post-Processing / Visual Presentation / Automation |
| **Platform** | Cross-platform (Python), dependencies on `numpy`, `Pillow`, and `scipy` |
| **Execution Type** | Standalone, run manually or scheduled; supports command-line arguments and a `--no-log` switch |

---

## Categories

- **Cosmetic Edge Smoothing** — Replaces jagged, staircase-edged opaque blobs with perfect, anti-aliased circles, for visual presentation purposes.
- **Data Preparation for Presentation, Not Analysis** — Produces a visually polished frame set intended for display/publication, explicitly not intended as input for further quantitative analysis or model training.
- **Automation / Scripting** — Fully automated, parameterized (`--alpha-cutoff`, `--supersample`, `--source-folder`, `--output-folder`, `--no-log`), no interactive input required.
- **Logging & Reporting** — Writes a timestamped, leveled log file capturing per-frame blob counts and outcomes.

---

## Script Purpose

The script takes transparent-background frames (each containing one or more jagged, staircase-edged opaque blobs) and replaces each blob with a perfect, anti-aliased circle, saving the result as new PNG images. It:

1. Resolves its own script directory and derives the project root one level up.
2. Accepts configurable source and output subfolders (`--source-folder`, default `04_Transparent`; `--output-folder`, default `05_Final_Rounded`), an alpha threshold (`--alpha-cutoff`, default `128`) defining what counts as an "opaque" data pixel, a supersampling factor (`--supersample`, default `4`) controlling edge smoothness, and an optional `--no-log` switch to disable file-based logging.
3. Creates the output folder if it does not already exist.
4. Scans the source folder for PNG files (transparency requires PNG; other formats are ignored).
5. For each frame, identifies every distinct opaque blob via its alpha channel, and for each blob:
   - Fits a circle based on the blob's bounding box (center and radius derived from its width/height).
   - Renders that circle's edge with anti-aliasing, using 4x (configurable) supersampling and downsampling for a smooth alpha gradient rather than a hard pixel step.
   - Fills any newly-exposed pixels — areas inside the new perfect circle that weren't part of the original jagged blob — using the color of the nearest originally-opaque pixel (nearest-neighbour inpainting).
6. Saves the resulting RGBA image to the output folder under the same filename.
7. Logs every step (`INFO`/`SUCCESS`/`WARNING`/`ERROR`) to `Logs\08_round_edges.log`, unless logging is disabled, and always echoes log entries to the console, including the number of blobs rounded per frame and a running total across the batch.
8. Continues past individual frame failures rather than stopping the whole batch, and reports a final summary (frames processed, blobs rounded, failures).

---

## Business / Process Purpose

This script performs a purely cosmetic finishing step for **Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks**, intended to make circular data regions look visually clean for presentation, without altering the underlying scientific data pipeline. Its business drivers include:

- **Polished visual output for review/publication**: Jagged, staircase-edged circular regions (a natural consequence of the coarse underlying measurement grid) can look unpolished in figures intended for presentations, reports, or publications. This script produces a visually smooth alternative for exactly those contexts.
- **Explicit separation from quantitative pipeline stages**: This script's edge-rounding is achieved by inventing pixel colors in the gap between the true jagged data edge and the fitted circle (via nearest-neighbour inpainting) — it does not increase the real resolution or accuracy of the underlying measurement. Keeping this as a distinct, clearly-labeled step (rather than folding it into earlier cleanup or mask-generation stages) prevents cosmetically-smoothed output from being mistaken for, or accidentally fed into, further quantitative analysis or model training, where fabricated boundary pixels would be undesirable.
- **Configurable smoothness/detection behavior**: Alpha threshold and supersampling factor are exposed as parameters, so the script can be retuned for different transparency conventions or visual quality/performance trade-offs without code changes.
- **Traceable processing**: Per-frame blob counts and a batch total are logged, giving a quick way to confirm the expected number of circular regions were detected and rounded in every frame.

---

## Logging the Result

Because this script's output is intended for visual review/presentation rather than further quantitative pipeline stages, its log primarily provides a processing record and basic sanity-check data (expected blob counts) rather than data provenance for model training.

- **Log file location**: `<ScriptDir>\Logs\08_round_edges.log` (created only when logging is enabled; disabled via `--no-log`)
- **Log format**: `[yyyy-MM-dd HH:mm:ss] [LEVEL] Message`
- **Log levels used**: `INFO`, `SUCCESS`, `WARNING`, `ERROR`
- **Log content captured**:
  - Script start/completion markers and script/project identification
  - Resolved source and output folder paths
  - The alpha cutoff and supersample factor used
  - A `WARNING`-level note when file logging has been disabled via `--no-log`
  - The number of PNG files found in the source folder
  - Per-frame processing result, including the number of blobs rounded in that frame
  - Any per-frame failures, without halting the rest of the batch
  - Final summary: frames processed, total blobs rounded across the batch, and failure count
- **Output artifact**: `<ProjectRoot>\Images\05_Final_Rounded\` (or the folder specified via `--output-folder`) — RGBA PNG frames with every detected blob replaced by a smooth, anti-aliased circle.
- **Console output**: Every log entry is echoed to the console in real time, regardless of whether file logging is enabled.
- **Research/ML pipeline value**: The per-frame blob count is a useful quick sanity check (e.g. confirming all 8 expected circular regions were detected in every frame) but this script's log is not intended as a data-provenance record in the way earlier pipeline stages' logs are, since its output should not be used as training/analysis input.

---

## Technology Assessment

| Aspect | Assessment |
|---|---|
| **Complexity** | Moderate. Combines connected-component blob detection, geometric circle fitting, supersampled anti-aliasing, and nearest-neighbour distance-transform inpainting, plus a reusable logging function and structured error capture. |
| **Dependencies** | Requires `numpy`, `Pillow` (PIL), and `scipy` (specifically `scipy.ndimage`). No external binaries or pretrained models required. |
| **Error Handling** | Moderate — explicit checks for source folder existence, per-frame try/except so one bad frame doesn't stop the batch, and a final failure count in the summary. Does not validate that input frames actually have meaningful transparency (an all-opaque or all-transparent frame would simply produce 0 or 1 detected blobs without raising an error). |
| **Portability** | Cross-platform (pure Python); depends on a specific relative folder layout (`Images`, `Logs`) matching the rest of the pipeline. No GPU or specialized hardware required. |
| **Idempotency** | Yes — re-running overwrites existing output frames and appends a new log entry; no destructive side effects on the source frames. |
| **Security Considerations** | Runs entirely locally against local image files. No network calls, no credential handling, no external executable dependency. |
| **Performance** | Light-to-moderate — supersampled anti-aliasing (default 4x) and per-pixel distance-transform inpainting add some computational cost relative to simple image operations, but processing remains fast (well under a second per typical frame) with no GPU required. |
| **Maintainability** | High — parameterized design (alpha cutoff, supersample factor, source/output folders, device-independent), a reusable `log()` function, and an explicit inline note on the cosmetic-only nature of this step make the script easy to understand, retune, and safely place within the broader pipeline. |

---

## Approval Considerations

- **Least privilege**: Requires read access to the source frames and write access to the output folder and (optionally) `Logs` — no elevated/administrative rights needed.
- **No third-party binary or model dependency**: Unlike several other scripts in this pipeline, this script has no external executable or pretrained model dependency to separately vet — only standard, widely-used Python libraries.
- **Circle-fitting geometry validation**: The fitted circle's center and radius are derived purely from each blob's bounding box (not a more precise fit such as a minimum-enclosing-circle or centroid-based method). Approvers should be aware that for non-circular or irregularly-shaped blobs, this bounding-box-based fit may produce a visually smooth but geometrically inaccurate circle relative to the blob's true extent — acceptable for this script's cosmetic purpose, but a reason this output should not be repurposed for measurement or analysis.
- **Data fidelity / do-not-reuse-downstream risk**: This is the most important approval consideration for this script. Its output pixels within the "filled gap" region between the true jagged edge and the fitted circle are fabricated via nearest-neighbour inpainting, not real measured data. Approvers should confirm that downstream consumers of `Images\05_Final_Rounded` (or the configured output folder) understand this output is for visual/presentation use only, and ensure it is not inadvertently substituted for the genuine cleaned data (e.g. `Images\03_Cleaned_lama`) in any further analysis or model-training step.
- **No external communication**: This script runs entirely locally against local image files; no network calls or data leave the machine.
- **No destructive actions**: Source frames are only read, never modified, moved, or deleted.
- **Auditability**: Structured, leveled logging with per-frame blob counts supports basic change-control and QA, though (as noted above) it is not intended as a data-provenance record in the same sense as earlier pipeline stages.
- **Path and environment assumptions**: Approvers should confirm the expected folder structure (`ProjectRoot\Images\<source-folder>`, `ProjectRoot\Images\<output-folder>`, `ScriptDir\Logs`) matches the target deployment environment.
- **Logging opt-out**: The `--no-log` flag allows disabling persistent logging entirely. Given this script's output is not a data-provenance-critical artifact, disabling logging here carries lower risk than doing so for earlier pipeline stages.
- **File overwrite behavior**: Re-running overwrites previous output frames without versioning — approvers may want to assess whether this matters, given the output's cosmetic-only intended use.

---

## Suggested Classification

**Classification: Low Risk / Cosmetic Post-Processing Utility — Output Not for Analytical Reuse**

- **Risk Level**: Low — read-only interaction with source frames, no network activity, no credential handling, no third-party binary or model dependency. The primary risk is not technical but process-related: the possibility of this script's cosmetically-altered output being mistaken for, or substituted for, genuine cleaned experimental data in a downstream analytical or model-training step.
- **Change Category**: Standard/pre-approved change candidate for routine or scheduled execution; no external dependency vetting required beyond standard Python library review.
- **Data Sensitivity**: Input frames are understood (from the broader project context) to be derived from scientific imaging data related to volumetric/mixing-tank experiments. This script's output frames contain a mix of genuine data pixels and fabricated (inpainted) boundary pixels:
  - Because part of each output frame's content is fabricated rather than measured, this output should be classified and labeled distinctly from genuine data products (e.g. `Images\03_Cleaned_lama`) in any data inventory or catalog, to prevent accidental reuse as if it were authentic experimental data.
  - The same storage/access sensitivity considerations that apply to the underlying experimental imaging data should still apply to this output, since it is visually derived from and closely resembles that data, even though its edge pixels are not genuine.
- **Recommended Handling**: Approve as a routine, low-risk cosmetic utility for presentation/publication purposes only. Ensure the distinction between this script's output and genuine cleaned data products is clearly documented and enforced in project data-management practice, so this step's fabricated-edge output is never inadvertently used as input to further quantitative analysis or model training.
