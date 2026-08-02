#!/bin/bash
# ============================================
# aiml_08_submit_gpu_training.sh
# SLURM batch script to run train_segmentation_demo.py on a university
# GPU cluster, for direct CPU vs GPU timing comparison.
#
# Most university HPC facilities use SLURM as their job scheduler. This
# script requests one GPU node, loads the required modules, and runs the
# same script used for the CPU benchmark - only --device changes.
#
# Usage (run from the login node of your university cluster):
#   sbatch aiml_08_submit_gpu_training.sh
#
# You WILL need to edit the #SBATCH lines below to match your specific
# university's cluster (partition/queue name, account/allocation code,
# and available module names) - check your university's HPC documentation
# or ask your research computing helpdesk for these exact values.
# ============================================

#SBATCH --job-name=segmentation_demo
#SBATCH --partition=gpu                 # <-- ask your HPC facility for the correct GPU partition/queue name
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00                 # 30 minute wall-clock limit, plenty for this small demo
#SBATCH --output=gpu_training_%j.log
#SBATCH --account=YOUR_ALLOCATION_HERE  # <-- your research group's allocation/account code

# Most clusters use environment modules to expose specific software versions.
# Exact module names vary by university - check `module avail` on your cluster.
module load python/3.11
module load cuda/12.1

# If you haven't already, create and populate a virtual environment once:
#   python -m venv ~/envs/seg-demo
#   source ~/envs/seg-demo/bin/activate
#   pip install torch torchvision opencv-python-headless numpy matplotlib
source ~/envs/seg-demo/bin/activate

# Report what GPU we actually got, for the log record
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Run the identical script used for the CPU benchmark, only --device differs,
# so timing comparisons are apples-to-apples.
python aiml_08_train_segmentation_demo.py \
    --epochs 15 \
    --lr 1e-3 \
    --batch-size 4 \
    --device cuda \
    --run-name lr1e-3_bs4_GPU

echo "Done. Compare results/lr1e-3_bs4_GPU_history.json against the CPU run's history.json."
