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
SSIM_LAMBDA=${12:-0.2}  # Default: 0.2 (tune for best PSNR)
OPACITY_REG=${13:-0.01}  # Default: 0.01 (tune for best PSNR)
SCALE_REG=${14:-0.01}    # Default: 0.01 
APP_EMDED_DIM=${15:-16}  # Default: 64 (tune for best PSNR)
SSIM_LAMBDA=${12:-0.2}        # Default: 0.2 (tune for best PSNR)
OPACITY_REG=${13:-0.01}       # Default: 0.01 (tune for best PSNR)
SCALE_REG=${14:-0.01}         # Default: 0.01 
APP_EMBED_DIM=${15:-16}       # Default: 16 (tune: 8, 16, 32, 64)
FEATURE_DIM=${16:-32}         # Default: 32 (tune: 16, 32, 64 when app_opt=ON)
GROW_GRAD2D=${17:-0.0008}     # Default: 0.0008 (for absgrad)
BILATERAL_SHAPE_X=${18:-16}   # Bilateral grid X dimension
BILATERAL_SHAPE_Y=${19:-16}   # Bilateral grid Y dimension
BILATERAL_SHAPE_W=${20:-8}    # Bilateral grid color dimension



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
CKPT=$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1))_rank0.pt
CHALLENGE_DIR=/disk/SN-NVS-2026-raw/${SCENE}-challenge


# -------------------------
# Memory & OOM Monitoring Setup
# -------------------------
mkdir -p ./logs
MEMORY_LOG="./logs/memory_${SLURM_JOB_ID}.csv"
OOM_LOG="./logs/oom_${SLURM_JOB_ID}.log"
PEAK_MEMORY_LOG="./logs/peak_memory_${SLURM_JOB_ID}.txt"
TRAINING_LOG=$RESULT_DIR/training_${SLURM_JOB_ID}.log

# Initialize memory logs
echo "timestamp,step,cpu_used_mb,cpu_available_mb,gpu_used_mb,gpu_total_mb,gpu_util_percent" > "$MEMORY_LOG"
echo "OOM Events Log - Job: $SLURM_JOB_ID" > "$OOM_LOG"

monitor_memory() {
    while true; do
        TIMESTAMP=$(date +%s.%N)
        
        # CPU Memory
        CPU_STATS=$(free -m | awk '/Mem:/ {print $3","$7}')
        
        # GPU Memory
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
        
        if [ ! -z "$GPU_MEM" ]; then
            echo "$TIMESTAMP,0,$CPU_STATS,$GPU_MEM" >> "$MEMORY_LOG"
        fi
        
        sleep 1
    done
}

check_oom_errors() {
    while true; do
        if grep -q "CUDA out of memory" "$TRAINING_LOG" 2>/dev/null; then
            echo "[$(date)] OOM detected in training log" >> "$OOM_LOG"
            # Log current memory state
            nvidia-smi >> "$OOM_LOG"
            echo "---" >> "$OOM_LOG"
        fi
        sleep 5
    done
}


echo "============================="
echo "🎯 MEMORY-OPTIMIZED CONFIGURATION"
echo "============================="
echo "Rasterize mode:      $RASTERIZE_MODE"
echo "DATA_FACTOR:         $DATA_FACTOR"
echo "PACKED_MODE:         $([ "$PACKED_MODE"     -eq 1 ] && echo ON || echo off)"
echo "BILATERAL_GRID:      $([ "$BILATERAL"       -eq 1 ] && echo ON || echo off)"
echo "APP_OPT:             $([ "$APP_OPT"         -eq 1 ] && echo ON || echo off)"
echo "ABSGRAD:             $([ "$ABSGRAD"         -eq 1 ] && echo ON || echo off)"
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

# Start memory monitors in background
monitor_memory & 
MONITOR_PID=$!

check_oom_errors &
OOM_CHECK_PID=$!

# --- TRAP TO CLEAN UP MONITORS ---
trap "kill $MONITOR_PID $OOM_CHECK_PID 2>/dev/null; analyze_memory" EXIT

echo "Memory monitors started with PIDs $MONITOR_PID $OOM_CHECK_PID"

analyze_memory() {
    if [ -f "$MEMORY_LOG" ]; then
        echo ""
        echo "===== FINAL MEMORY ANALYSIS ====="
        PEAK_GPU=$(awk -F',' 'NR>1 {print $5}' "$MEMORY_LOG" | sort -nr | head -1)
        PEAK_CPU=$(awk -F',' 'NR>1 {print $3}' "$MEMORY_LOG" | sort -nr | head -1)
        AVG_GPU=$(awk -F',' 'NR>1 {sum+=$5; count++} END {print int(sum/count)}' "$MEMORY_LOG")
        
        echo "Peak GPU Memory:    ${PEAK_GPU}MB"
        echo "Peak CPU Memory:    ${PEAK_CPU}MB"
        echo "Average GPU Memory: ${AVG_GPU}MB"
        
        echo "Peak GPU: ${PEAK_GPU}MB" > "$PEAK_MEMORY_LOG"
        echo "Peak CPU: ${PEAK_CPU}MB" >> "$PEAK_MEMORY_LOG"
        echo "Average GPU: ${AVG_GPU}MB" >> "$PEAK_MEMORY_LOG"
        echo "Timestamp: $(date)" >> "$PEAK_MEMORY_LOG"
    fi
}

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
    \
    --batch_size 1 \
    \
    $FLAGS 

# --ssim_lambda 0.2 \
# --strategy.grow_grad2d 0.0008 \

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
PACKED_MODE:             $([ "$PACKED_MODE"     -eq 1 ] && echo ON || echo off)  
BILATERAL_GRID:          $([ "$BILATERAL"       -eq 1 ] && echo ON || echo off)  
APP_OPT:                 $([ "$APP_OPT"         -eq 1 ] && echo ON || echo off) 
ABSGRAD:                 $([ "$ABSGRAD"         -eq 1 ] && echo ON || echo off) 

================================================================================
CONFIGURATION
================================================================================
Rasterize mode:          $RASTERIZE_MODE
ANTIALIASED:             $([ "$ANTIALIASED"     -eq 1 ] && echo ON || echo off)
DENSIFICATION_STRATEGY:  $DENSIFICATION_STRATEGY
SCENE:                   $SCENE
VERSION:                 $VERSION
MAX_STEPS:               $MAX_STEPS
DATA_FACTOR:             $DATA_FACTOR

Your Config:
packed=ON, absgrad=ON, app_opt=$([ "$APP_OPT" -eq 1 ] && echo ON || echo off), bilateral=$([ "$BILATERAL" -eq 1 ] && echo ON || echo off)
Estimated Total:         
================================================================================
EOL

echo "✅ Run config saved to $LOG_FILE"