#!/bin/bash
#SBATCH --job-name=soccernet_2dgs
#SBATCH --output=./logs/train_2dgs_%j.out
#SBATCH --account=soccernet_nvs
#SBATCH --error=./logs/error_2dgs_%j.log
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time 02:00:00

set -e

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/hensemberk/dev/Soccernet/gsplat"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

if [ $# -lt 2 ]; then
    print_error "Not enough arguments"
    echo "Usage: SCENE DENSIFICATION [FLAGS]"
    echo "  SCENE: Scene name (e.g., soccernet)"
    echo ""
    echo "Flags:"
    echo "  --data-factor N         Downsample factor (default: 2)"
    echo "  --max-steps N           Max steps (default: 40000)"
    echo "  --colmap-dir PATH       Path to COLMAP sparse folder"
    echo "  --data-dir PATH         Path to dataset"
    echo "  --depth-loss            Enable depth loss with DA3"
    echo "  --depth-ground          Enable ground-plane prior"
    echo "  --use-controller        Enable PPISP controller"
    echo "  --ssim-lambda N         SSIM weight (default: 0.2)"
    echo "  --wandb-steps N         W&B eval frequency (default: 1000)"
    exit 1
fi

SCENE=$1
shift 1


# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# 2DGS-specific features
DEPTH_LOSS=false
DEPTH_GROUND=false
NORMAL_LOSS=false
DIST_LOSS=false
ABSGRAD=false
APP_OPT=false


# Hyperparameters
DATA_FACTOR=2
MAX_STEPS=40000
MAX_REFINE_STEPS=25000
OPACITY_REG=0.01
SCALE_REG=0.01
SSIM_LAMBDA=0.2

# Paths
DATA_DIR="/disk/SN-NVS-2026-raw/${SCENE}"
COLMAP_DIR="/disk/SN-NVS-2026-raw/${SCENE}/sparse/0"
CHALLENGE_DIR="/disk/SN-NVS-2026-raw/${SCENE}-challenge"
GROUND_DIR="/disk/SN-NVS-2026-raw/${SCENE}/mask/masks"
DEPTH_DIR="${DATA_DIR}/dae3/DA3MONO-LARGE/depth_maps.npz"

# W&B config
USE_WANDB=true
WANDB_STEPS_EVAL=1000
WANDB_RUN_NAME=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth-loss)
            DEPTH_LOSS=true
            shift
            ;;
        --absgrad)
            ABSGRAD=true
            shift
            ;;
        --app-opt)
            APP_OPT=true
            shift
            ;;
        --data-factor)
            DATA_FACTOR=$2
            shift 2
            ;;
        --max-steps)
            MAX_STEPS=$2
            shift 2
            ;;
        --max-refine)
            MAX_REFINE_STEPS=$2
            shift 2
            ;;
        --opacity-reg)
            OPACITY_REG=$2
            shift 2
            ;;
        --scale-reg)
            SCALE_REG=$2
            shift 2
            ;;
        --ssim-lambda)
            SSIM_LAMBDA=$2
            shift 2
            ;;
        --data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --normal-loss) 
            NORMAL_LOSS=$2
            shift 2
            ;;
        --dist-loss) 
            DIST_LOSS=$2
            shift 2
            ;;
        --colmap-dir)
            COLMAP_DIR=$2
            shift 2
            ;;
        --wandb-steps)
            WANDB_STEPS_EVAL=$2
            shift 2
            ;;
        *)
            print_warning "Unknown flag: $1"
            shift
            ;;
    esac
done

# ============================================================================
# BUILD FEATURE NAME & OUTPUT DIRECTORIES
# ============================================================================

FEATURES=()

if [ "$DEPTH_LOSS" = true ]; then
    FEATURES+=("depthloss")
fi

if [ "$DEPTH_GROUND" = true ]; then
    FEATURES+=("groundprior")
fi

# Always add 2DGS to feature name
if [ ${#FEATURES[@]} -eq 0 ]; then
    FEATURE_NAME="2dgs_baseline"
else
    FEATURE_NAME="2dgs_$(IFS=_; echo "${FEATURES[*]}")"
fi

CONFIG_SUFFIX=""
if [ "$MAX_STEPS" != "40000" ]; then
    CONFIG_SUFFIX="${CONFIG_SUFFIX}_s${MAX_STEPS}"
fi

OUTPUT_BASE_DIR="${CHALLENGE_DIR}/${SCENE}/${FEATURE_NAME}${CONFIG_SUFFIX}"

# Handle multiple runs
if [ -d "$OUTPUT_BASE_DIR" ]; then
    i=1
    while [ -d "${OUTPUT_BASE_DIR}_run${i}" ]; do
        ((i++))
    done
    OUTPUT_DIR="${OUTPUT_BASE_DIR}_run${i}"
    RUN_NUM=$i
else
    OUTPUT_DIR="$OUTPUT_BASE_DIR"
    RUN_NUM=0
fi

RESULT_DIR="/tmp/gsplat_2dgs_train_${SCENE}_${DENSIFICATION}_${FEATURE_NAME}${CONFIG_SUFFIX}_run${RUN_NUM}"
WANDB_RUN_NAME="${SCENE}_2dgs_${DENSIFICATION}_${FEATURE_NAME}_run${RUN_NUM}"
WANDB_PATH_CHALLENGE="${CHALLENGE_DIR}/sparse/0"



cleanup_on_failure() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "----------------------------------------------------"
        case $exit_code in
            1)   echo "❌ ERROR 1: General error or Python Crash (Import/Logic)" ;;
            130) echo "❌ ERROR 130: Script terminated by User (Ctrl+C)" ;;
            137) echo "❌ ERROR 137: Out of Memory (OOM) - Job killed by Slurm" ;;
            139) echo "❌ ERROR 139: Segmentation Fault (C++/CUDA core error)" ;;
            *)   echo "❌ ERROR $exit_code: Unknown crash" ;;
        esac

        echo "⚠️ Cleaning up folders..."
        [ -d "$RESULT_DIR" ] && rm -rf "$RESULT_DIR" && echo "🗑️ Deleted temp dir: $RESULT_DIR"
        [ -d "$OUTPUT_DIR" ] && rm -rf "$OUTPUT_DIR"  && echo "🗑️ Deleted output dir: $OUTPUT_DIR"
        echo "----------------------------------------------------"
    fi
}

trap cleanup_on_failure EXIT

mkdir -p "$RESULT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

LOG_FILE="$OUTPUT_DIR/experiment_log.txt"
CONFIG_FILE="$OUTPUT_DIR/config.txt"

# ============================================================================
# LOG CONFIGURATION
# ============================================================================

{
    print_header "2DGS EXPERIMENT CONFIGURATION"

    echo ""
    echo "📋 EXPERIMENT IDENTITY"
    printf "%-30s %s\n" "Scene:" "$SCENE"
    printf "%-30s %s\n" "Model:" "2DGS (2D Gaussian Splatting)"
    printf "%-30s %s\n" "Densification:" "$DENSIFICATION"
    printf "%-30s %s\n" "Features:" "$FEATURE_NAME"
    printf "%-30s %s\n" "Run Number:" "$RUN_NUM"

    echo ""
    echo "🔧 TRAINING PARAMETERS"
    printf "%-30s %s\n" "Max Steps:" "$MAX_STEPS"
    printf "%-30s %s\n" "Data Factor:" "$DATA_FACTOR"
    printf "%-30s %s\n" "Batch Size:" "1"
    printf "%-30s %s\n" "SSIM Lambda:" "$SSIM_LAMBDA"

    echo ""
    echo "🎯 2DGS-SPECIFIC FEATURES"
    [ "$DEPTH_LOSS" = true ]     && printf "%-30s %s\n" "Depth Loss (DA3):" "✓ ON" || printf "%-30s %s\n" "Depth Loss (DA3):" "✗ OFF"
    [ "$DEPTH_GROUND" = true ]   && printf "%-30s %s\n" "Ground Plane Prior:" "✓ ON" || printf "%-30s %s\n" "Ground Plane Prior:" "✗ OFF"
    [ "$USE_CONTROLLER" = true ] && printf "%-30s %s\n" "PPISP Controller:" "✓ ON" || printf "%-30s %s\n" "PPISP Controller:" "✗ OFF"

    echo ""
    echo "📊 HYPERPARAMETERS"
    printf "%-30s %s\n" "Opacity Regularization:" "$OPACITY_REG"
    printf "%-30s %s\n" "Scale Regularization:" "$SCALE_REG"

    echo ""
    echo "📂 PATHS"
    printf "%-30s %s\n" "Data Directory:" "$DATA_DIR"
    printf "%-30s %s\n" "COLMAP Directory:" "$COLMAP_DIR"
    printf "%-30s %s\n" "Temp Train Directory:" "$RESULT_DIR"
    printf "%-30s %s\n" "Output Directory:" "$OUTPUT_DIR"

    echo ""
    echo "⏱️ TIMING"
    printf "%-30s %s\n" "Start Time:" "$(date '+%Y-%m-%d %H:%M:%S')"

} | tee "$LOG_FILE"

cp "$LOG_FILE" "$CONFIG_FILE"

print_info "Configuration logged to: $LOG_FILE"

# ============================================================================
# BUILD TRAINING FLAGS
# ============================================================================

FLAGS=""
FLAGS="$FLAGS --disable_viewer"
FLAGS="$FLAGS --data_factor $DATA_FACTOR"
FLAGS="$FLAGS --opacity_reg $OPACITY_REG"
FLAGS="$FLAGS --scale_reg $SCALE_REG"
FLAGS="$FLAGS --max_steps $MAX_STEPS"
FLAGS="$FLAGS --ssim_lambda $SSIM_LAMBDA"


# FLAGS="$FLAGS --wandb_run_name $WANDB_RUN_NAME"
# FLAGS="$FLAGS --wandb_steps $WANDB_STEPS_EVAL"
# FLAGS="$FLAGS --max_refine_steps $MAX_REFINE_STEPS"
# FLAGS="$FLAGS --wandb_path_challenge $WANDB_PATH_CHALLENGE"

# Add 2DGS-specific flags

if [ "$ABSGRAD" = true ]; then
    FLAGS="$FLAGS --absgrad"
fi

if [ "$DEPTH_LOSS" = true ]; then
    FLAGS="$FLAGS --depth_loss"
    FLAGS="$FLAGS --mini_depth_dir $DEPTH_DIR"
fi

if [ "$DIST_LOSS" = true ]; then
    FLAGS="$FLAGS --dist_loss"
fi

if [ "$NORMAL_LOSS" = true ]; then
    FLAGS="$FLAGS --normal_loss"
fi



print_header "GPU ENVIRONMENT CHECK"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate soccernet

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

srun python3 -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ Compute capability: {torch.cuda.get_device_capability()}')
"

# ============================================================================
# TRAINING WITH 2DGS
# ============================================================================

print_header "STARTING 2DGS TRAINING"
print_info "Scene: $SCENE"
print_info "Model: 2D Gaussian Splatting"
print_info "Densification: $DENSIFICATION"
print_info "Features: $FEATURE_NAME"
print_info "Temp output: $RESULT_DIR"

START_TIME=$(date +%s)

# Use simple_trainer_2dgs.py instead of simple_trainer.py
srun python "$REPO_ROOT/examples/simple_trainer_2dgs.py" \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --colmap_dir "$COLMAP_DIR" \
    --no-normalize_world_space \
    --test_every 0 \
    --save_steps 10000 20000 30000 40000 \
    $FLAGS \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "Training exit code: $TRAIN_EXIT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ============================================================================
# POST-TRAINING: EVAL + COPY CKPT + CLEANUP
# ============================================================================

print_header "TRAINING COMPLETED"
print_info "Duration: $((DURATION / 60)) minutes ($DURATION seconds)"
print_info "Model: 2D Gaussian Splatting"

CKPT="$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1)).pt"

if [ -f "$CKPT" ]; then
    print_info "Checkpoint found: $(basename $CKPT)"

    # ── 1. Run evaluation ──
    print_header "RUNNING 2DGS EVALUATION"

    srun python "$REPO_ROOT/examples/eval_challenge.py" \
        --ckpt "$CKPT" \
        --data_dir "$CHALLENGE_DIR" \
        --result_folder "$OUTPUT_DIR" \
        --specific \
        2>&1 | tee -a "$LOG_FILE"

    # ── 2. Copy checkpoint ──
    CKPT_DEST_DIR="$OUTPUT_DIR/ckpts"
    mkdir -p "$CKPT_DEST_DIR"
    cp "$CKPT" "$CKPT_DEST_DIR/"
    print_info "Checkpoint copied to: $CKPT_DEST_DIR/$(basename $CKPT)"

    # ── 3. Cleanup temp directory ──
    if [ $TRAIN_EXIT -eq 0 ]; then
        echo "🧹 Removing temp training directory: $RESULT_DIR"
        rm -rf "$RESULT_DIR" && echo "✓ Temp directory removed" || echo "⚠ Failed to remove temp dir"
    fi

    # ── 3. Delete the temporary training directory ──
    rm -rf "$RESULT_DIR"
    print_info "Temp training directory removed: $RESULT_DIR"


else
    print_warning "Checkpoint not found at: $CKPT"
    print_warning "Skipping eval and cleanup."
fi

# ============================================================================
# GENERATE SUMMARY
# ============================================================================

SUMMARY_FILE="$OUTPUT_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# 2D Gaussian Splatting (2DGS) Experiment Summary

## Configuration
- **Scene**: $SCENE
- **Model**: 2D Gaussian Splatting (Surfel-based)
- **Densification**: $DENSIFICATION
- **Features**: $FEATURE_NAME
- **Run**: ${RUN_NUM}

## 2DGS-Specific Features
- **Depth Loss (DA3)**: $DEPTH_LOSS
- **Ground Plane Prior**: $DEPTH_GROUND
- **PPISP Controller**: $USE_CONTROLLER

## Parameters
- **Max Steps**: $MAX_STEPS
- **Data Factor**: $DATA_FACTOR
- **Opacity Reg**: $OPACITY_REG
- **Scale Reg**: $SCALE_REG
- **SSIM Lambda**: $SSIM_LAMBDA

## Results
- **Start Time**: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')
- **End Time**: $(date '+%Y-%m-%d %H:%M:%S')
- **Duration**: $((DURATION / 60)) minutes
- **Output Dir**: $OUTPUT_DIR
- **Checkpoint**: $CKPT_DEST_DIR/$(basename $CKPT)

## Model Details
2D Gaussian Splatting represents scenes using 2D Gaussian surfels lying on
ground planes, rather than 3D Gaussians floating in space. This is particularly
suited for:
- Ground-plane constrained scenes (sports fields, streets)
- Normal estimation (floor detection, planar surfaces)
- Efficient geometry supervision

## Paths
- Data: $DATA_DIR
- COLMAP: $COLMAP_DIR
- Challenge: $CHALLENGE_DIR
- Depth Dir: $DEPTH_DIR

## References
- 2DGS Paper: https://github.com/hbb1/2d-gaussian-splatting
- gsplat 2DGS Implementation: https://github.com/nerfstudio-project/gsplat
EOF

print_info "Summary saved to: $SUMMARY_FILE"

echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
    print_header "✅ 2DGS TRAINING FINISHED SUCCESSFULLY"
else
    print_header "❌ 2DGS TRAINING FAILED (exit code: $TRAIN_EXIT)"
fi