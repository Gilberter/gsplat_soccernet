#!/bin/bash
#SBATCH --job-name=sbatch_soccernet
#SBATCH --output=./logs/colmap_rendering_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_rendering_%j.log 
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu 
#SBATCH --mem=10G
#SBATCH --time=01:00:00

# Create logs directory if it doesn't exist
mkdir -p ./logs

echo "Running job: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Initialize and activate Conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate soccernet

# Fix common libstdc++ issues by preloading the conda version
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Verify environment and GPU access
srun python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability()}')
print(f'Arch list: {torch.cuda.get_arch_list()}')
"

RESULT_FOLDER="/disk/SN-NVS-2026-raw/results-soccernet/scene-1-scene-1-ultimate-8"
CKPT="$RESULT_FOLDER/ckpts/ckpt_29999_rank0.pt"
CHALLENGE_DIR="/disk/SN-NVS-2026-raw/scene-1-challenge"

# Execute the evaluation script
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/eval_challenge.py \
    --ckpt "$CKPT" \
    --data_dir "$CHALLENGE_DIR" \
    --result_folder "$RESULT_FOLDER"
