#!/bin/bash
#SBATCH --job-name=difix_nvs
#SBATCH --output=./logs/difix_%j.out
#SBATCH --error=./logs/difix_%j.err
#SBATCH --account=soccernet_nvs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Difix3D distillation recipe: resume from a converged 80k MCMC+antialiased+
# app_opt checkpoint and finetune for a short distillation stage. Scene name
# is user-supplied; checkpoint is looked up from the per-scene map below.
#
# Usage:
#   sbatch examples/train_difix.sbatch            # defaults to scene-2
#   sbatch examples/train_difix.sbatch scene-3    # override
SCENE=${1:-scene-2}

declare -A CKPTS=(
    [scene-1]="/disk/SN-NVS-2026-raw/scene-1-challenge/results-depth-prior/mcmc/antialiased_appopt_s80000_run5/ckpts/ckpt_079999_rank0.pt"
    [scene-2]="/disk/SN-NVS-2026-raw/scene-2-challenge/results-depth-prior/mcmc/antialiased_appopt_s80000_run2/ckpts/ckpt_079999_rank0.pt"
    [scene-3]="/disk/SN-NVS-2026-raw/scene-3-challenge/results-depth-prior/mcmc/antialiased_appopt_s80000_run1/ckpts/ckpt_079999_rank0.pt"
    [scene-4]="/disk/SN-NVS-2026-raw/scene-4-challenge/results-depth-prior/mcmc/antialiased_appopt_s80000/ckpts/ckpt_079999_rank0.pt"
    [scene-5]="/disk/SN-NVS-2026-raw/scene-5-challenge/results-depth-prior/mcmc/antialiased_appopt_s80000/ckpts/ckpt_079999_rank0.pt"
)

CKPT="${CKPTS[$SCENE]}"
if [ -z "$CKPT" ]; then
    echo "No checkpoint mapping for scene: $SCENE"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/examples"

# Resume at step 80000 and finetune for 12k more steps with Difix distillation.
# wandb_training.sh forces --antialiased and --app-opt when --difix is set.
bash "$SCRIPT_DIR/wandb_training.sh" "$SCENE" mcmc \
    --difix \
    --ckpt "$CKPT" \
    --data-factor 2 \
    --max-steps 92000 \
    --wandb-steps 1000 \
    --ssim-lambda 0.2
