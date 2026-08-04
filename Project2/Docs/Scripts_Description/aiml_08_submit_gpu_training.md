# Script Description: aiml_08_submit_gpu_training.sh

## Overview

| Field | Value |
| :--- | :--- |
| **Script Name** | `aiml_08_submit_gpu_training.sh` |
| **Language** | Bash / SLURM Batch Script |
| **Project** | Project 2: AIML-Driven Super-Resolution and Volumetric Reconstruction for Mixing Tanks |
| **Category** | High-Performance Computing (HPC) Submission / Hardware Benchmarking / GPU Acceleration |
| **Platform** | Linux-based HPC Clusters (SLURM Workload Manager) |
| **Execution Type** | Batch Job Submission via SLURM (`sbatch aiml_08_submit_gpu_training.sh`) |

---

## Categories

1. **HPC Workload Management & Job Scheduling**  
   Configures resource allocation requests (GPUs, CPU cores, memory, wall-time limits) for submitting deep learning training jobs to high-performance computing clusters via SLURM.

2. **Hardware Benchmarking & Performance Profiling**  
   Facilitates direct CPU vs. GPU training time and convergence comparisons by executing identical model training parameters on CUDA hardware.

3. **Environment & Hardware Inspection**  
   Automates module loading (`cuda`, `python`), virtual environment activation, and target GPU diagnostic logging using `nvidia-smi`.

---

## Script Purpose

The `aiml_08_submit_gpu_training.sh` shell script orchestrates automated GPU training job execution for the mixing tank segmentation model (`aiml_08_train_segmentation_demo.py`) on university HPC clusters running SLURM. 

### Key Features & Execution Flow

* **SLURM Resource Allocation (`#SBATCH` Directives):**
  * Requests a single GPU unit (`--gres=gpu:1`) within designated cluster partitions (`--partition=gpu`).
  * Allocates 4 CPU cores (`--cpus-per-task=4`) and 16 GB system memory (`--mem=16G`) to handle data loading and preprocessing.
  * Sets a maximum runtime window of 30 minutes (`--time=00:30:00`) and standardizes job output logs (`gpu_training_%j.log`, where `%j` expands to the unique job ID).
* **Environment Provisioning:**
  * Dynamically loads required software modules (Python 3.11, CUDA 12.1) using the cluster environment module system (`module load`).
  * Activates a pre-configured Python virtual environment (`~/envs/seg-demo`) containing PyTorch CUDA builds and computer vision dependencies[cite: 8].
* **Hardware Diagnostic Profiling:**
  * Queries and records allocated GPU device details (GPU name, VRAM capacity, driver version) via `nvidia-smi` before training starts[cite: 8].
* **Controlled Training Execution:**
  * Invokes `aiml_08_train_segmentation_demo.py` with identical hyperparameters as CPU benchmark runs (15 epochs, learning rate $1\times 10^{-3}$, batch size 4), explicitly directing compute to PyTorch's `cuda` backend[cite: 8].

---

## Business / Process Purpose

Executing deep learning training on CPU hardware often introduces severe performance bottlenecks during multi-epoch optimization and spatial tensor convolutions.

`aiml_08_submit_gpu_training.sh` addresses these bottlenecks by:
1. **Ensuring Rigorous Performance Benchmarking:** Enforcing controlled variable testing where hyperparameter settings remain identical across CPU and GPU hardware runs, isolating compute device acceleration factors[cite: 8].
2. **Streamlining Cluster Integration:** Standardizing execution templates for university HPC clusters, allowing researchers to dispatch compute-heavy training tasks off local workstations[cite: 8].
3. **Optimizing Compute Allocation:** Standardizing cluster queue requests to ensure efficient queue placement without over-allocating cluster node resources[cite: 8].

---

## Directives & Parameters

### SLURM Header Directives (`#SBATCH`)

| Directive | Value | Purpose |
| :--- | :--- | :--- |
| `--job-name` | `segmentation_demo` | Defines the display name of the job in SLURM queue tools (`squeue`)[cite: 8]. |
| `--partition` | `gpu` | Specifies target GPU hardware queue partition (requires local HPC adjustment)[cite: 8]. |
| `--gres` | `gpu:1` | Requests 1 physical GPU card on the worker node[cite: 8]. |
| `--cpus-per-task` | `4` | Allocates 4 CPU worker threads for PyTorch `DataLoader` workers[cite: 8]. |
| `--mem` | `16G` | Reserves 16 GB RAM on the host node[cite: 8]. |
| `--time` | `00:30:00` | Sets a strict job execution wall-time limit (HH:MM:SS)[cite: 8]. |
| `--output` | `gpu_training_%j.log` | Sets output log filename, where `%j` is replaced by the job ID[cite: 8]. |
| `--account` | `YOUR_ALLOCATION_HERE` | User research allocation code for billing/quota tracking[cite: 8]. |

### Script Execution Parameters

| Command Argument | Parameter Value | Description |
| :--- | :--- | :--- |
| `--epochs` | `15` | Total training epochs[cite: 8]. |
| `--lr` | `1e-3` | Learning rate for the Adam optimizer[cite: 8]. |
| `--batch-size` | `4` | Batch size per GPU compute step[cite: 8]. |
| `--device` | `cuda` | Hardware target for PyTorch tensor operations[cite: 8]. |
| `--run-name` | `lr1e-3_bs4_GPU` | Experiment identifier string used for export filenames[cite: 8]. |

---

## Environment & Prerequisites Setup

Before submitting the job to SLURM, standard virtual environment initialization is required on the HPC login node:

```bash
# Create target environment
python -m venv ~/envs/seg-demo

# Activate environment
source ~/envs/seg-demo/bin/activate

# Install machine learning dependencies
pip install torch torchvision opencv-python-headless numpy matplotlib