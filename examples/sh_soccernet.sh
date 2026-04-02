#!/bin/bash
#SBATCH --job-name=sbatch_soccernet
#SBATCH --output=./logs/colmap_rendering_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_rendering_%j.log 
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
BILATERAL=${4:-0}        # Default: OFF (memory intensive!)
ANTIALIASED=${5:-0}      # Default: OFF (hurts PSNR)
ABSGRAD=${6:-1}          # Default: ON (minimal memory impact)
DENSIFICATION=${7:-0}    # 0=default, 1=mcmc
APP_OPT=${8:-0}          # Default: OFF (memory intensive!)
MAX_STEPS=${9:-40000}
PACKED_MODE=${10:-1}     # Default: ON (reduces memory)
DATA_FACTOR=${11:-1}     # Default: 1 (full resolution)


if [ "$RASTER_MODE" -eq 1 ]; then
    RASTERIZE_MODE="antialiased"
else
    RASTERIZE_MODE="classic"
fi

if [ "$DENSIFICATION" -eq 1 ]; then
    DENSIFICATION_STRATEGY="mcmc"
else
    DENSIFICATION_STRATEGY="default"
fi

FLAGS=""
[ "$BILATERAL"      -eq 1 ] && FLAGS="$FLAGS --bilateral_grid_fused --post_processing bilateral_grid"
[ "$ANTIALIASED"    -eq 1 ] && FLAGS="$FLAGS --antialiased"
[ "$ABSGRAD"        -eq 1 ] && FLAGS="$FLAGS --strategy.absgrad"
[ "$APP_OPT"        -eq 1 ] && FLAGS="$FLAGS --app_opt"
[ "$PACKED_MODE"    -eq 1 ] && FLAGS="$FLAGS --packed"


# -------------------------
# Paths
# -------------------------

DATA_DIR=/disk/SN-NVS-2026-raw/${SCENE}
RESULT_DIR=/disk/SN-NVS-2026-raw/results-soccernet/${SCENE}-${VERSION}
COLMAP_DIR=/disk/SN-NVS-2026-raw/${SCENE}/sparse/0
CKPT=$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1))_rank0.pt
CHALLENGE_DIR=/disk/SN-NVS-2026-raw/${SCENE}-challenge

echo "============================="
echo "🎯 MEMORY-OPTIMIZED CONFIGURATION"
echo "============================="
echo "Rasterize mode:      $RASTERIZE_MODE"
echo "DATA_FACTOR:         $DATA_FACTOR"
echo "PACKED_MODE:         $([ "$PACKED_MODE"     -eq 1 ] && echo ON || echo off)  [MEMORY SAVER]"
echo "BILATERAL_GRID:      $([ "$BILATERAL"       -eq 1 ] && echo ON || echo off)  [HIGH MEMORY]"
echo "APP_OPT:             $([ "$APP_OPT"         -eq 1 ] && echo ON || echo off)  [MEDIUM MEMORY]"
echo "ABSGRAD:             $([ "$ABSGRAD"         -eq 1 ] && echo ON || echo off)  [MINIMAL MEMORY]"
echo "ANTIALIASED:         $([ "$ANTIALIASED"     -eq 1 ] && echo ON || echo off)"
echo "DENSIFICATION:       $DENSIFICATION_STRATEGY"
echo "============================="
echo "SCENE:               $SCENE"
echo "VERSION:             $VERSION"
echo "MAX_STEPS:           $MAX_STEPS"
echo "DATA_DIR:            $DATA_DIR"
echo "RESULT_DIR:          $RESULT_DIR"
echo "MEMORY_LOG:          $MEMORY_LOG"
echo "GPU_LOG:             $GPU_LOG"
echo "============================="

monitor_usage() {
    echo "Time, RAM_Used_MB, GPU_Used_MB, GPU_Util_%" > "./logs/mem_stats_$SLURM_JOB_ID.csv"
    while true; do
        RAM=$(free -m | awk '/Mem:/ {print $3}')
        GPU=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        echo "$(date +%H:%M:%S), $RAM, $GPU" >> "./logs/mem_stats_$SLURM_JOB_ID.csv"
        sleep 1
    done
}

# Start monitor in background
monitor_usage & 
MONITOR_PID=$!

# --- TRAP TO CLEAN UP MONITOR ---
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "Memory monitor started with PID $MONITOR_PID"

# -------------------------
# Training with MEMORY OPTIMIZATIONS
# -------------------------
echo "Starting training at $(date)"
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/simple_trainer.py $DENSIFICATION_STRATEGY \
    --max_steps $MAX_STEPS \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --save_steps $MAX_STEPS \
    --data_factor $DATA_FACTOR \
    --no-normalize_world_space \
    --no-load_exposure \
    --test_every 0 \
    --colmap_dir $COLMAP_DIR \
    --rasterize_mode $RASTERIZE_MODE \
    --disable_viewer \
    \
    --sh_degree 3 \
    --opacity_reg 0.01 \
    --scale_reg 0.01 \
    --strategy.grow_grad2d 0.0008 \
    \
    --batch_size 1 \
    --ssim_lambda 0.2 \
    \
    $FLAGS 

echo "Training completed at $(date) with exit code"

# -------------------------
# Evaluation
# -------------------------
srun python /home/hensemberk/dev/Soccernet/gsplat/examples/eval_challenge.py \
    --ckpt $CKPT \
    --data_dir $CHALLENGE_DIR \
    --result_folder "results-${VERSION}-1"


LOG_FILE=$CHALLENGE_DIR/results-${VERSION}-1/run_config.txt

echo "Saving run configuration to $LOG_FILE"

mkdir -p "$(dirname "$LOG_FILE")"

cat <<EOL > $LOG_FILE
================================================================================
SLURM JOB CONFIGURATION
================================================================================
SLURM_JOB_ID:            $SLURM_JOB_ID
NODE:                    $SLURM_NODELIST

================================================================================
MEMORY OPTIMIZATION FLAGS
================================================================================
PACKED_MODE:             $([ "$PACKED_MODE"     -eq 1 ] && echo ON || echo off)  [Reduces memory 30-40%]
BILATERAL_GRID:          $([ "$BILATERAL"       -eq 1 ] && echo ON || echo off)  [+5-10GB VRAM]
APP_OPT:                 $([ "$APP_OPT"         -eq 1 ] && echo ON || echo off)  [+2-3GB VRAM]
ABSGRAD:                 $([ "$ABSGRAD"         -eq 1 ] && echo ON || echo off)  [Minimal overhead]

================================================================================
CONFIGURATION
================================================================================
Rasterize mode:          $RASTERIZE_MODE
ANTIALIASED:             $([ "$ANTIALIASED"     -eq 1 ] && echo ON || echo off)
WITH_UT:                 $([ "$WITH_UT"         -eq 1 ] && echo ON || echo off)
WITH_EVAL3D:             $([ "$WITH_EVAL3D"     -eq 1 ] && echo ON || echo off)
DENSIFICATION_STRATEGY:  $DENSIFICATION_STRATEGY
SCENE:                   $SCENE
VERSION:                 $VERSION
MAX_STEPS:               $MAX_STEPS
DATA_FACTOR:             1 (Full resolution)

================================================================================
MEMORY USAGE ESTIMATES (GB)
================================================================================
Base (no options):       2-3 GB
+ packed mode:           -1 GB (reduces to 1-2 GB)
+ absgrad:               0.5 GB
+ app_opt:               +2-3 GB
+ bilateral_grid:        +3-5 GB
+ antialiased:           +0.5 GB

Your Config:
packed=ON, absgrad=ON, app_opt=$([ "$APP_OPT" -eq 1 ] && echo ON || echo off), bilateral=$([ "$BILATERAL" -eq 1 ] && echo ON || echo off)
Estimated Total:         ~3-8 GB (varies by scene resolution)
================================================================================
EOL

echo "✅ Run config saved to $LOG_FILE"