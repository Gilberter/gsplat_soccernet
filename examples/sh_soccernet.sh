#!/bin/bash
#SBATCH --job-name=sbatch_soccernet
#SBATCH --output=./logs/colmap_rendering_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_rendering_%j.log 
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu 
#SBATCH --mem=25G
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
BILATERAL=${4:-0}        # Default: OFF (memory intensive!)
ANTIALIASED=${5:-0}      # Default: OFF (hurts PSNR)
ABSGRAD=${6:-1}          # Default: ON (minimal memory impact)
DENSIFICATION=${7:-0}    # 0=default, 1=mcmc
APP_OPT=${8:-0}          # Default: OFF (memory intensive!)
MAX_STEPS=${9:-40000}
PACKED_MODE=${10:-1}     # Default: ON (reduces memory)
DATA_FACTOR=${11:-1}     # Default: 1 (full resolution)

# MORE SETTINGS (tune for best PSNR, but may increase memory):

SSIM_LAMBDA=${SSIM_LAMBDA:-0.2}        # Default: 0.2 (tune for best PSNR)
OPACITY_REG=${OPACITY_REG:-0.01}       # Default: 0.01 (tune for best PSNR)
SCALE_REG=${SCALE_REG:-0.01}         # Default: 0.01 
APP_EMBED_DIM=${APP_EMBED_DIM:-16}       # Default: 16 (tune: 8, 16, 32, 64)
FEATURE_DIM=${FEATURE_DIM:-32}         # Default: 32 (tune: 16, 32, 64 when app_opt=ON)
GROW_GRAD2D=${GROW_GRAD2D:-0.0008}     # Default: 0.0008 (for absgrad)
BILATERAL_SHAPE_X=${BILATERAL_SHAPE_X:-16}   # Bilateral grid X dimension
BILATERAL_SHAPE_Y=${BILATERAL_SHAPE_Y:-16}   # Bilateral grid Y dimension
BILATERAL_SHAPE_W=${BILATERAL_SHAPE_W:-8}    # Bilateral grid color dimension



if [ "$RASTER_MODE" -eq 1 ]; then
    RASTERIZE_MODE="antialiased"
else
    RASTERIZE_MODE="classic"
fi

if [ "$DENSIFICATION" -eq 1 ]; then
    DENSIFICATION_STRATEGY="mcmc"
else
    DENSIFICATION_STRATEGY="default --strategy.grow_grad2d $GROW_GRAD2D"
fi

# Post-processing mode selection
if [ "$POST_PROCESSING" -eq 1 ]; then
    POST_PROC_MODE="bilateral_grid"
    POST_PROC_FLAG="--bilateral_grid_fused --post_processing bilateral_grid --bilateral_grid_shape $BILATERAL_SHAPE_X $BILATERAL_SHAPE_Y $BILATERAL_SHAPE_W"
elif [ "$POST_PROCESSING" -eq 2 ]; then
    POST_PROC_MODE="ppisp"
    POST_PROC_FLAG="--post_processing ppisp --ppisp_use_controller true --ppisp_controller_distillation true"
else
    POST_PROC_MODE="none"
    POST_PROC_FLAG=""
fi

FLAGS=""
[ "$BILATERAL"      -eq 1 ] && FLAGS="$FLAGS --bilateral_grid_fused --post_processing bilateral_grid"
[ "$ANTIALIASED"    -eq 1 ] && FLAGS="$FLAGS --antialiased"
[ "$ABSGRAD"        -eq 1 ] && FLAGS="$FLAGS --strategy.absgrad --absgrad"
[ "$APP_OPT"        -eq 1 ] && FLAGS="$FLAGS --app_opt --app_embed_dim $APP_EMBED_DIM"
[ "$PACKED_MODE"    -eq 1 ] && FLAGS="$FLAGS --packed"


# -------------------------
# Paths
# -------------------------

DATA_DIR=/disk/SN-NVS-2026-raw/${SCENE}
RESULT_DIR=/disk/SN-NVS-2026-raw/results-soccernet/${SCENE}-${VERSION}
COLMAP_DIR=/disk/SN-NVS-2026-raw/${SCENE}/sparse/0
CHALLENGE_DIR=/disk/SN-NVS-2026-raw/${SCENE}-challenge

BASE_DIR="$CHALLENGE_DIR/${SCENE}-${VERSION}"
RESULT_BASE=/disk/SN-NVS-2026-raw/results-soccernet/${SCENE}-${VERSION}


if [ -d "$RESULT_BASE" ]; then
    i=1
    while [ -d "${RESULT_BASE}_run$i" ]; do
        ((i++))
    done
    RESULT_DIR="${RESULT_BASE}_run$i"
    OUTPUT_DIR="${BASE_DIR}_run$i"
else
    RESULT_DIR="$RESULT_BASE"
    OUTPUT_DIR="$BASE_DIR"
fi

cleanup_on_failure() {
    local exit_code=$?
    # If the exit code is NOT 0 (meaning it crashed)
    if [ $exit_code -ne 0 ]; then
        echo "❌ Script failed with exit code $exit_code. Cleaning up folders..."
        [ -d "$RESULT_DIR" ] && rm -rf "$RESULT_DIR"
        [ -d "$OUTPUT_DIR" ] && rm -rf "$OUTPUT_DIR"
        echo "🗑️ Deleted: $RESULT_DIR and $OUTPUT_DIR"
    fi
}

trap cleanup_on_failure EXIT

mkdir -p "$RESULT_DIR"

mkdir -p "$OUTPUT_DIR"


CKPT=$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1))_rank0.pt

LOG_FILE_TRAIN=$RESULT_DIR/run_config.txt


{
echo "============================================================"
echo "⚙️ CONFIGURATION"
echo "============================================================"

printf "%-25s %s\n" "Scene:" "$SCENE"
printf "%-25s %s\n" "Version:" "$VERSION"
printf "%-25s %s\n" "Max steps:" "$MAX_STEPS"
printf "%-25s %s\n" "Data factor:" "$DATA_FACTOR"

echo "------------------------------------------------------------"

printf "%-25s %s\n" "Raster mode:" "$RASTERIZE_MODE"
printf "%-25s %s\n" "Packed mode:" "$([ "$PACKED_MODE" -eq 1 ] && echo ON || echo OFF)"
printf "%-25s %s\n" "Absgrad:" "$([ "$ABSGRAD" -eq 1 ] && echo ON || echo OFF)"
printf "%-25s %s\n" "App opt:" "$([ "$APP_OPT" -eq 1 ] && echo ON || echo OFF)"
printf "%-25s %s\n" "Bilateral:" "$([ "$BILATERAL" -eq 1 ] && echo ON || echo OFF)"
printf "%-25s %s\n" "Antialiased:" "$([ "$ANTIALIASED" -eq 1 ] && echo ON || echo OFF)"

echo "------------------------------------------------------------"

printf "%-25s %s\n" "Opacity reg:" "$OPACITY_REG"
printf "%-25s %s\n" "Scale reg:" "$SCALE_REG"
printf "%-25s %s\n" "App embed dim:" "$APP_EMBED_DIM"

echo "============================================================"

echo "============================================================"
echo "📂 PATHS"
echo "============================================================"

printf "%-20s %s\n" "DATA_DIR:" "$DATA_DIR"
printf "%-20s %s\n" "RESULT_DIR:" "$RESULT_DIR"
printf "%-20s %s\n" "COLMAP_DIR:" "$COLMAP_DIR"
printf "%-20s %s\n" "CKPT:" "$CKPT"

echo "============================================================"

echo "============================================================"
echo "🏋️ START TRAINING"
echo "============================================================"

} >> "$LOG_FILE_TRAIN"
START_TIME=$(date +%s)
# -------------------------
# Training with MEMORY OPTIMIZATIONS
# -------------------------
{
echo "Starting training at $(date)"
} >> "$LOG_FILE_TRAIN"

srun python /home/hensemberk/dev/Soccernet/gsplat/examples/simple_trainer.py $DENSIFICATION_STRATEGY \
    --max_steps $MAX_STEPS \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --save_steps $MAX_STEPS \
    --data_factor $DATA_FACTOR \
    --no-normalize_world_space \
    --save_steps 10000 20000 30000 40000 \
    --no-load_exposure \
    --test_every 0 \
    --colmap_dir $COLMAP_DIR \
    --rasterize_mode $RASTERIZE_MODE \
    --feature_dim $FEATURE_DIM \
    --disable_viewer \
    \
    --sh_degree 3 \
    --opacity_reg $OPACITY_REG \
    --scale_reg $SCALE_REG \
    \
    --batch_size 1 \
    \
    $FLAGS 

# --ssim_lambda 0.2 \
# 


END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
echo "============================================================"
echo "✅ TRAINING FINISHED"
echo "⏱️ Duration: $DURATION seconds (~$((DURATION/60)) min)"
echo "📅 End time: $(date)"
echo "============================================================"

echo "Training completed at $(date) with exit code"
} >> "$LOG_FILE_TRAIN"

# -------------------------
# Evaluation
# -------------------------
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/eval_challenge.py \
    --ckpt $CKPT \
    --data_dir $CHALLENGE_DIR \
    --result_folder "$(basename $OUTPUT_DIR)"


LOG_FILE=$OUTPUT_DIR/run_conf.txt

echo "Saving run configuration to $LOG_FILE"

mkdir -p "$(dirname "$LOG_FILE")"

cat <<EOL > "$LOG_FILE"
================================================================================
SLURM JOB INFO
================================================================================
Job ID:                $SLURM_JOB_ID
Node:                  $SLURM_NODELIST
User:                  $(whoami)
Start Time:            $(date -d @$START_TIME 2>/dev/null || date)
End Time:              $(date)
Duration (sec):        $DURATION

================================================================================
ENVIRONMENT
================================================================================
Conda Env:             $CONDA_DEFAULT_ENV
Python:                $(which python)
CUDA Visible Devices:  $CUDA_VISIBLE_DEVICES

================================================================================
DATA PATHS
================================================================================
DATA_DIR:              $DATA_DIR
RESULT_DIR:            $RESULT_DIR
COLMAP_DIR:            $COLMAP_DIR
CHALLENGE_DIR:         $CHALLENGE_DIR
CKPT:                  $CKPT

================================================================================
MAIN CONFIGURATION
================================================================================
Scene:                 $SCENE
Version:               $VERSION
Max Steps:             $MAX_STEPS
Data Factor:           $DATA_FACTOR

================================================================================
TRAINING FLAGS
================================================================================
Raster Mode:           $RASTERIZE_MODE
Packed Mode:           $([ "$PACKED_MODE" -eq 1 ] && echo ON || echo OFF)
AbsGrad:               $([ "$ABSGRAD" -eq 1 ] && echo ON || echo OFF)
App Opt:               $([ "$APP_OPT" -eq 1 ] && echo ON || echo OFF)
Bilateral Grid:        $([ "$BILATERAL" -eq 1 ] && echo ON || echo OFF)
Antialiased:           $([ "$ANTIALIASED" -eq 1 ] && echo ON || echo OFF)

Densification:         $DENSIFICATION_STRATEGY

================================================================================
HYPERPARAMETERS
================================================================================
Opacity Reg:           $OPACITY_REG
Scale Reg:             $SCALE_REG
SSIM Lambda:           $SSIM_LAMBDA
App Embed Dim:         $APP_EMBED_DIM
Feature Dim:           $FEATURE_DIM
Grow Grad2D:           $GROW_GRAD2D

Bilateral Shape X:     $BILATERAL_SHAPE_X
Bilateral Shape Y:     $BILATERAL_SHAPE_Y
Bilateral Shape W:     $BILATERAL_SHAPE_W

================================================================================
FULL COMMAND (REPRODUCIBILITY)
================================================================================
python simple_trainer.py $DENSIFICATION_STRATEGY \
    --max_steps $MAX_STEPS \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --data_factor $DATA_FACTOR \
    --colmap_dir $COLMAP_DIR \
    --rasterize_mode $RASTERIZE_MODE \
    $FLAGS

================================================================================
EOL

echo "✅ Run config saved to $LOG_FILE"