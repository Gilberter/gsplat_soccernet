#!/bin/bash
#SBATCH --job-name=sbatch_soccernet
#SBATCH --output=./logs/scene_prior%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/scene_prior%j.log 
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu 
#SBATCH --mem=40G
#SBATCH --time 01:00:00

echo "Running job: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"


source /opt/miniforge3/etc/profile.d/conda.sh
conda activate soccernet

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

srun python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability()}')
print(torch.cuda.get_arch_list())
"

# -------------------------
# Arguments
# -------------------------

SCENE=$1
VERSION=$2
RASTER_MODE=${3:-0}      # 0 classic, 1 antialiased
BILATERAL=${4:-0}
ANTIALIASED=${5:-0}
WITH_UT=${6:-0}
WITH_EVAL3D=${7:-0}
ABSGRAD=${8:-0}
DENSIFICATION=${9:-0}
MASKS_DIR=$10
GROUND_DEPTH_LOSS=$11
DEPTH_LOSS=$12
COLMAP_D=$13
MAX_STEPS=${14:-30000}

if [ "$RASTER_MODE" -eq 1 ]; then
    RASTERIZE_MODE="antialiased"
else
    RASTERIZE_MODE="classic"
fi

if ["$DENSIFICATION" -eq 1]; then
    DENSIFICATION_STRATEGY="mcmc"
else
    DENSIFICATION_STRATEGY="default"
fi

FLAGS=""
[ "$BILATERAL"   -eq 1 ] && FLAGS="$FLAGS --bilateral_grid_fused"
[ "$ANTIALIASED" -eq 1 ] && FLAGS="$FLAGS --antialiased"
[ "$WITH_UT"     -eq 1 ] && FLAGS="$FLAGS --with_ut"
[ "$WITH_EVAL3D" -eq 1 ] && FLAGS="$FLAGS --with_eval3d"
[ "$ABSGRAD"     -eq 1 ] && FLAGS="$FLAGS --absgrad"
[ "$GROUND_DEPTH_LOSS"     -eq 1 ] && FLAGS="$FLAGS --ground_depth_loss"
[ "$DEPTH_LOSS"     -eq 1 ] && FLAGS="$FLAGS --depth_loss"


# -------------------------
# Paths
# -------------------------

DATA_DIR=/disk/SN-NVS-2026-raw/${SCENE}
RESULT_DIR=/disk/SN-NVS-2026-raw/results-soccernet/${SCENE}-${VERSION}


COLMAP_DIR=/disk/SN-NVS-2026-raw/${SCENE}/sparse/0
CKPT=$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1))_rank0.pt
CHALLENGE_DIR=/disk/SN-NVS-2026-raw/${SCENE}-challenge

echo "============================="
echo "Rasterize mode: $RASTERIZE_MODE"
echo "BILATERAL:      $([ "$BILATERAL"   -eq 1 ] && echo ON || echo off)"
echo "ANTIALIASED:    $([ "$ANTIALIASED" -eq 1 ] && echo ON || echo off)"
echo "WITH_UT:        $([ "$WITH_UT"     -eq 1 ] && echo ON || echo off)"
echo "WITH_EVAL3D:    $([ "$WITH_EVAL3D" -eq 1 ] && echo ON || echo off)"
echo "ABSGRAD:        $([ "$ABSGRAD"     -eq 1 ] && echo ON || echo off)"
echo "FLAGS:          [${FLAGS}]"
echo "SCENE:          $SCENE"
echo "VERSION:        $VERSION"
echo "MAX_STEPS:      $MAX_STEPS"
echo "DATA_DIR:       $DATA_DIR"
echo "RESULT_DIR:     $RESULT_DIR"
echo "COLMAP_DIR:     $COLMAP_DIR"
echo "CKPT:           $CKPT"
echo "============================="

# -------------------------
# Training
# -------------------------
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/simple_trainer.py $DENSIFICATION_STRATEGY \
    --max_steps $MAX_STEPS \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --eval_steps $MAX_STEPS \
    --save_steps $MAX_STEPS \
    --data_factor 1 \
    --no-normalize_world_space \
    --no-load_exposure \
    --test_every 0 \
    --colmap_dir $COLMAP_DIR \
    --rasterize_mode $RASTERIZE_MODE \
    --disable_viewer \
    $FLAGS 
    


# -------------------------
# Evaluation
# -------------------------
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/eval_challenge.py \
    --ckpt $CKPT \
    --data_dir $CHALLENGE_DIR \
    --result_folder "results-${VERSION}-1"


LOG_FILE=$CHALLENGE_DIR/results-${VERSION}-1/run_config.txt

echo "Saving run configuration to $LOG_FILE"

cat <<EOL > $LOG_FILE
SLURM_JOB_ID:   $SLURM_JOB_ID
NODE:            $SLURM_NODELIST

Rasterize mode: $RASTERIZE_MODE
BILATERAL:      $([ "$BILATERAL"   -eq 1 ] && echo ON || echo off)
ANTIALIASED:    $([ "$ANTIALIASED" -eq 1 ] && echo ON || echo off)
WITH_UT:        $([ "$WITH_UT"     -eq 1 ] && echo ON || echo off)
WITH_EVAL3D:    $([ "$WITH_EVAL3D" -eq 1 ] && echo ON || echo off)
ABSGRAD:        $([ "$ABSGRAD"     -eq 1 ] && echo ON || echo off)
FLAGS:          [${FLAGS}]
SCENE:          $SCENE
VERSION:        $VERSION
MAX_STEPS:      $MAX_STEPS
DATA_DIR:       $DATA_DIR
RESULT_DIR:     $RESULT_DIR
COLMAP_DIR:     $COLMAP_DIR
CKPT:           $CKPT
EOL