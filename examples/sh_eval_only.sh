#!/bin/bash
#SBATCH --job-name=sbatch_soccernet
#SBATCH --output=./logs/colmap_rendering_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_rendering_%j.log 
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu 
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Create logs directory if it doesn't exist
mkdir -p ./logs

echo "Running job: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Initialize and activate Conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate soccernet

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

SCENE=$1
VERSION=$2
CKPT_RANK=${3:-290000}  # Default checkpoint rank to evaluate

VERSION_NAME=$(basename "$VERSION")
echo "$VERSION_NAME"

CHALLENGE_DIR="/disk/SN-NVS-2026-raw/${SCENE}-challenge"
CKPT_DIR="/disk/SN-NVS-2026-raw/results-soccernet/${VERSION_NAME}/ckpts/ckpt_${CKPT_RANK}_rank0.pt"
RESULT_DIR=""

RESULT_BASE_DIR="${CHALLENGE_DIR}/${VERSION_NAME}"

if [ -d "$RESULT_BASE_DIR" ]; then
    RESULT_DIR=${RESULT_BASE_DIR}/results-${CKPT_RANK}
else
    RESULT_DIR="${CHALLENGE_DIR}/${VERSION_NAME}"
fi

echo "Target Result Folder: $RESULT_DIR"
echo "Target Checkpoint: $CKPT_DIR"

# Execute the evaluation script
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/eval_challenge.py \
    --ckpt "$CKPT_DIR" \
    --data_dir "$CHALLENGE_DIR" \
    --result_folder "$RESULT_DIR" \
    --specific
