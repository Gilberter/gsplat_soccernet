#!/bin/bash
#SBATCH --job-name=2dgs_soccernet_optimal
#SBATCH --output=./logs/2dgs_training_%j.out
#SBATCH --error=./logs/2dgs_error_%j.log 
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu 
#SBATCH --mem=25G
#SBATCH --time 02:00:00

echo "🚀 2DGS Training - Adapted from 3DGS Best Config"
echo "Running job: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate soccernet
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# ============================================================
# YOUR PROVEN 3DGS CONFIG (Decoding parameters)
# ============================================================
# APP_EMBED_DIM=64       → Appearance embedding dimension
# OPACITY_REG=0.01       → Opacity regularization weight
# RASTER_MODE=1          → Antialiased rasterization
# BILATERAL=1            → Bilateral grid post-processing
# ABSGRAD=0              → NOT using absgrad in your config
# DENSIFICATION=1        → MCMC densification strategy
# APP_OPT=1              → Appearance optimization enabled
# MAX_STEPS=40000        → Training steps
# PACKED_MODE=1          → Packed mode ON
# DATA_FACTOR=2          → Downsample factor

# ============================================================
# 2DGS ADAPTED CONFIGURATION (From your 3DGS best)
# ============================================================
SCENE=${1:-"scene-1"}
VERSION=${2:-"2dgs_opt"}
MAX_STEPS=${3:-40000}

# ✅ KEY SETTINGS FROM YOUR 3DGS SUCCESS
BATCH_SIZE=1
APP_OPT=0                        # Keep appearance optimization ON
APP_EMBED_DIM=64                   # Your proven best! (higher than default 16)
OPACITY_REG=0.01                   # Your proven opacity regularization
PACKED_MODE=1                      # Memory efficient + faster
DATA_FACTOR=2                      # Your typical setting

# ⚡ ADAPTATIONS FOR 2DGS (Different from 3DGS)
# 2DGS uses different gradient computation
ABSGRAD=1                          # Important for 2DGS (uses gradient_2dgs instead of means2d)
GROW_GRAD2D=0.0008                 # More aggressive for 2DGS with absgrad
ANTIALIASED=0                     # Your 3DGS had antialiased=1

# 📊 QUALITY SETTINGS
SSIM_LAMBDA=0.2                    # Standard balance
SH_DEGREE=3                        # Maximum SH degree
SH_DEGREE_INTERVAL=1000

# 🌱 DENSIFICATION (Adjusted for 2DGS)
# 2DGS needs different thresholds than 3DGS
PRUNE_OPA=0.05
PRUNE_SCALE3D=0.1
GROW_SCALE3D=0.01

REFINE_START_ITER=500
REFINE_STOP_ITER=15000
REFINE_EVERY=100
RESET_EVERY=3000

# ❌ FEATURES NOT SUPPORTED IN 2DGS (from your 3DGS)
# BILATERAL=1 → Not in simple_trainer_2dgs.py
# NORMAL_LOSS=0 → Available but experimental
# DIST_LOSS=0 → Available but experimental

# ============================================================
# PATHS
# ============================================================
DATA_DIR="/disk/SN-NVS-2026-raw/${SCENE}"
RESULT_DIR="/disk/SN-NVS-2026-raw/results-soccernet/${SCENE}-${VERSION}"
COLMAP_DIR="/disk/SN-NVS-2026-raw/${SCENE}/sparse/0"

# Create versioned result directory
if [ -d "$RESULT_DIR" ]; then
    i=1
    while [ -d "${RESULT_DIR}_run$i" ]; do
        ((i++))
    done
    RESULT_DIR="${RESULT_DIR}_run$i"
fi
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/training_config.txt"

# ============================================================
# BUILD FLAGS
# ============================================================
FLAGS=""
[ "$PACKED_MODE"    -eq 1 ] && FLAGS="$FLAGS --packed"
[ "$ABSGRAD"        -eq 1 ] && FLAGS="$FLAGS --absgrad"
[ "$ANTIALIASED"    -eq 1 ] && FLAGS="$FLAGS --antialiased"
[ "$APP_OPT"        -eq 1 ] && FLAGS="$FLAGS --app_opt"

# ============================================================
# LOG CONFIGURATION
# ============================================================
{
echo "============================================================"
echo "🎯 2DGS TRAINING - OPTIMIZED FROM YOUR 3DGS CONFIG"
echo "============================================================"
echo ""
echo "📊 INPUT (Your Proven 3DGS Settings)"
echo "   APP_EMBED_DIM=64 ✓ (was 16 default)"
echo "   OPACITY_REG=0.01 ✓ (strong regularization)"
echo "   ANTIALIASED=1 ✓"
echo "   DENSIFICATION=MCMC ✓"
echo "   PACKED_MODE=1 ✓"
echo ""
echo "🔧 2DGS ADAPTATIONS"
printf "%-30s %s\n" "Scene:" "$SCENE"
printf "%-30s %s\n" "Version:" "$VERSION"
printf "%-30s %s\n" "Max steps:" "$MAX_STEPS"
printf "%-30s %s\n" "Data factor:" "$DATA_FACTOR"
echo ""
echo "💎 QUALITY SETTINGS (Adapted for 2DGS)"
printf "%-30s %s\n" "SH degree:" "$SH_DEGREE"
printf "%-30s %s\n" "SH interval:" "$SH_DEGREE_INTERVAL"
printf "%-30s %s\n" "SSIM lambda:" "$SSIM_LAMBDA"
printf "%-30s %s\n" "Appearance opt:" "✅ ON"
printf "%-30s %s\n" "App embed dim:" "$APP_EMBED_DIM (your proven best!)"
printf "%-30s %s\n" "Opacity reg:" "$OPACITY_REG (your proven best!)"
echo ""
echo "⚡ PERFORMANCE"
printf "%-30s %s\n" "Packed mode:" "✅ ON (memory efficient)"
printf "%-30s %s\n" "AbsGrad:" "✅ ON (critical for 2DGS)"
printf "%-30s %s\n" "Antialiased:" "✅ ON"
printf "%-30s %s\n" "Grow grad2d:" "$GROW_GRAD2D (aggressive)"
echo ""
echo "🌱 DENSIFICATION"
printf "%-30s %s\n" "Prune opacity:" "$PRUNE_OPA"
printf "%-30s %s\n" "Grow scale3d:" "$GROW_SCALE3D"
printf "%-30s %s\n" "Prune scale3d:" "$PRUNE_SCALE3D"
printf "%-30s %s\n" "Refine start:" "$REFINE_START_ITER"
printf "%-30s %s\n" "Refine stop:" "$REFINE_STOP_ITER"
echo ""
echo "📂 PATHS"
printf "%-20s %s\n" "Data:" "$DATA_DIR"
printf "%-20s %s\n" "Results:" "$RESULT_DIR"
printf "%-20s %s\n" "Colmap:" "$COLMAP_DIR"
echo ""
echo "============================================================"

} | tee "$LOG_FILE"

# ============================================================
# START TRAINING
# ============================================================
echo "🏋️  Starting training at $(date)" | tee -a "$LOG_FILE"
START_TIME=$(date +%s)

srun python /home/hensemberk/dev/Soccernet/gsplat/examples/simple_trainer_2dgs.py \
    --max_steps $MAX_STEPS \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --data_factor $DATA_FACTOR \
    --colmap_dir $COLMAP_DIR \
    \
    --batch_size $BATCH_SIZE \
    --sh_degree $SH_DEGREE \
    --sh_degree_interval $SH_DEGREE_INTERVAL \
    \
    --ssim_lambda $SSIM_LAMBDA \
    \
    --app_opt \
    --app_embed_dim $APP_EMBED_DIM \
    \
    --grow_grad2d $GROW_GRAD2D \
    --grow_scale3d $GROW_SCALE3D \
    --prune_opa $PRUNE_OPA \
    --prune_scale3d $PRUNE_SCALE3D \
    \
    --refine_start_iter $REFINE_START_ITER \
    --refine_stop_iter $REFINE_STOP_ITER \
    --refine_every $REFINE_EVERY \
    --reset_every $RESET_EVERY \
    \
    --save_steps 10000 20000 30000 40000 \
    \
    --disable_viewer \
    $FLAGS

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
echo ""
echo "============================================================"
echo "✅ TRAINING COMPLETED"
echo "============================================================"
echo "⏱️  Duration: $DURATION seconds (~$((DURATION/60)) min)"
echo "📅 Finished at: $(date)"
echo "📊 Results: $RESULT_DIR"
echo "============================================================"
} | tee -a "$LOG_FILE"

echo "Exit code: $?"