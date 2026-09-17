#!/bin/bash
#SBATCH --job-name=dgtd_spp
#SBATCH --partition=gpu             # <-- set to your cluster's GPU partition
#SBATCH --gres=gpu:1                # 1 GPU is enough (<1 GB needed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=dgtd_%j.out
#SBATCH --error=dgtd_%j.err

set -euo pipefail

module purge
# --- adjust to your cluster's module names. If you installed cuda-toolkit
#     INTO the conda env (recommended below), you can drop this entirely. ---
module load cuda/12.2
# --------------------------------------------------------------------------

# --- conda activation -------------------------------------------------------
# `conda activate` needs the shell hook in a non-interactive batch job;
# plain `conda activate` will fail with "shell not properly configured".
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dgtd

# If `conda` itself is not on PATH in batch jobs, hard-code the base instead:
#   source /home/$USER/miniconda3/etc/profile.d/conda.sh
#   conda activate dgtd
# ---------------------------------------------------------------------------

echo "host: $(hostname)"
echo "conda env: ${CONDA_DEFAULT_ENV}   python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import cupy; print('cupy', cupy.__version__, cupy.cuda.runtime.getDeviceCount(), 'device(s)')"

SCRATCH=${SLURM_SUBMIT_DIR}/out_${SLURM_JOB_ID}
mkdir -p "$SCRATCH"

srun python ho_solver_gpu.py paper_mesh.msh \
     --total-fs 25.0 \
     --outdir "$SCRATCH" \
     --snap-fs 0.25 \
     --cfl 0.3

# post-process on the same node (CPU work, cheap)
python postprocess.py "$SCRATCH"

echo "results in $SCRATCH"
