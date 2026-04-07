#!/bin/bash
#SBATCH --job-name=gsplat_experiment
#SBATCH --output=./logs/train_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_%j.log
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --mem=25G
#SBATCH --time 02:00:00

################################################################################
# GSPLAT Training Script with Organized Experiment Structure
# ============================================================================
# Usage:
#   bash train_experiment.sh SCENE DENSIFICATION [FLAGS]
#
# Arguments:
#   SCENE           : Scene name (required)
#   DENSIFICATION   : 'classic' or 'mcmc' (required)
#   FLAGS:
#     --absgrad                Enable absolute gradients
#     --antialiased            Enable antialiased rasterization
#     --app-opt                Enable appearance optimization
#     --bilateral              Enable bilateral grid post-processing
#     --data-factor N          Downsample factor (default: 1)
#     --max-steps N            Max training steps (default: 40000)
#     --colmap-dir PATH        Path to COLMAP sparse folder
#     --data-dir PATH          Path to dataset
#
# Examples:
#   bash train_experiment.sh soccernet classic --absgrad --bilateral
#   bash train_experiment.sh garden mcmc --bilateral --data-factor 2
################################################################################

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
    echo -e "${RED}��${NC} $1"
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
    echo "  DENSIFICATION: 'classic' or 'mcmc'"
    echo ""
    echo "Flags:"
    echo "  --absgrad               Enable absolute gradients"
    echo "  --antialiased           Enable antialiased rasterization"
    echo "  --app-opt               Enable appearance optimization"
    echo "  --post-proccesing {bilateral_grid,ppisp}      Enable bilateral grid post-processing"
    echo "  --data-factor N         Downsample factor (default: 1)"
    echo "  --max-steps N           Max steps (default: 40000)"
    echo "  --colmap-dir PATH       Path to COLMAP sparse folder"
    echo "  --data-dir PATH         Path to dataset"
    echo "  --packed                Enable Packed Mode"
    exit 1
fi

SCENE=$1
DENSIFICATION=$2
shift 2

# Validate densification type
if [[ ! "$DENSIFICATION" =~ ^(classic|mcmc)$ ]]; then
    print_error "DENSIFICATION must be 'classic' or 'mcmc', got: $DENSIFICATION"
    exit 1
fi

# ============================================================================
# DEFAULT PARAMETERS (Can be overridden by flags)
# ============================================================================

# Feature flags
ABSGRAD=false
ANTIALIASED=false
APP_OPT=false
BILATERAL=false
PPISP=false
PACKED=false

# Hyperparameters
DATA_FACTOR=2
MAX_STEPS=40000
OPACITY_REG=0.01
SCALE_REG=0.01
APP_EMBED_DIM=16
FEATURE_DIM=32
GROW_GRAD2D=0.0008

# Bilateral grid parameters
BILATERAL_SHAPE_X=16
BILATERAL_SHAPE_Y=16
BILATERAL_SHAPE_W=8

# Paths (set defaults, can be overridden)
DATA_DIR="/disk/SN-NVS-2026-raw/${SCENE}"
COLMAP_DIR="/disk/SN-NVS-2026-raw/${SCENE}/sparse/0"
CHALLENGE_DIR="/disk/SN-NVS-2026-raw/${SCENE}-challenge"

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --absgrad)
            ABSGRAD=true
            shift
            ;;
        --antialiased)
            ANTIALIASED=true
            shift
            ;;
        --app-opt)
            APP_OPT=true
            shift
            ;;
        --post-processing)
            case $2 in
                bilateral*) # Matches bilateral or bilateral_grid
                    BILATERAL=true ;;
                ppisp)
                    PPISP=true ;;
                *)
                    echo "Warning: Unknown post-processing method '$2'" ;;
            esac
            shift 2
            ;;
        --data-factor)
            DATA_FACTOR=$2
            shift 2
            ;;
        --max-steps)
            MAX_STEPS=$2
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
        --data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --colmap-dir)
            COLMAP_DIR=$2
            shift 2
            ;;
        *)
            print_warning "Unknown flag: $1"
            shift
            ;;
    esac
done

# ============================================================================
# GENERATE OUTPUT FOLDER NAME (Meaningful & Organized)
# ============================================================================

# Build folder structure: DENSIFICATION / FEATURES / SPECIFIC_CONFIG
FEATURES=()

## ADD FEATURES TO ADD TO THE NAME OF THE FOLDER
if [ "$ABSGRAD" = true ]; then
    FEATURES+=("absgrad")
fi

if [ "$ANTIALIASED" = true ]; then
    FEATURES+=("antialiased")
fi

if [ "$APP_OPT" = true ]; then
    FEATURES+=("appopt")
fi

if [ "$PPISP" = true ]; then
    FEATURES+=("ppisp")
fi

if [ "$BILATERAL" = true ]; then
    FEATURES+=("bilateral")
fi


# If no features, use baseline
if [ ${#FEATURES[@]} -eq 0 ]; then
    FEATURES=("baseline")
fi

FEATURE_NAME=$(IFS=_; echo "${FEATURES[*]}")

# Generate configuration suffix based on non-default parameters
CONFIG_SUFFIX=""
if [ "$MAX_STEPS" != "40000" ]; then
    CONFIG_SUFFIX="${CONFIG_SUFFIX}_s${MAX_STEPS}"
fi

# Final result directory structure
RESULT_BASE_DIR="/disk/SN-NVS-2026-raw/results-soccernet/${SCENE}/${DENSIFICATION}/${FEATURE_NAME}${CONFIG_SUFFIX}"
OUTPUT_BASE_DIR="${CHALLENGE_DIR}/${SCENE}/${DENSIFICATION}/${FEATURE_NAME}${CONFIG_SUFFIX}"

# Handle multiple runs (run1, run2, etc.)
if [ -d "$RESULT_BASE_DIR" ]; then
    i=1
    while [ -d "${RESULT_BASE_DIR}_run${i}" ]; do
        ((i++))
    done
    RESULT_DIR="${RESULT_BASE_DIR}_run${i}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}_run${i}"
    RUN_NUM=$i
else
    RESULT_DIR="$RESULT_BASE_DIR"
    OUTPUT_DIR="$OUTPUT_BASE_DIR"
    RUN_NUM=0
fi

mkdir -p "$RESULT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

# ============================================================================
# LOG CONFIGURATION
# ============================================================================

LOG_FILE="$RESULT_DIR/experiment_log.txt"
CONFIG_FILE="$RESULT_DIR/config.txt"

{
    print_header "EXPERIMENT CONFIGURATION"
    
    echo ""
    echo "📋 EXPERIMENT IDENTITY"
    printf "%-30s %s\n" "Scene:" "$SCENE"
    printf "%-30s %s\n" "Densification:" "$DENSIFICATION"
    printf "%-30s %s\n" "Feature Set:" "$FEATURE_NAME"
    printf "%-30s %s\n" "Run Number:" "$RUN_NUM"
    
    echo ""
    echo "🔧 TRAINING PARAMETERS"
    printf "%-30s %s\n" "Max Steps:" "$MAX_STEPS"
    printf "%-30s %s\n" "Data Factor:" "$DATA_FACTOR"
    printf "%-30s %s\n" "Batch Size:" "1"
    printf "%-30s %s\n" "SH Degree:" "3"
    
    echo ""
    echo "🎯 FEATURES ENABLED"
    [ "$ABSGRAD" = true ] && printf "%-30s %s\n" "Absgrad:" "✓ ON" || printf "%-30s %s\n" "Absgrad:" "✗ OFF"
    [ "$ANTIALIASED" = true ] && printf "%-30s %s\n" "Antialiased:" "✓ ON" || printf "%-30s %s\n" "Antialiased:" "✗ OFF"
    [ "$APP_OPT" = true ] && printf "%-30s %s\n" "App Opt:" "✓ ON" || printf "%-30s %s\n" "App Opt:" "✗ OFF"
    [ "$BILATERAL" = true ] && printf "%-30s %s\n" "Bilateral Grid:" "✓ ON" || printf "%-30s %s\n" "Bilateral Grid:" "✗ OFF"
    [ "$PPISP" = true ] && printf "%-30s %s\n" "PPISP:" "✓ ON" || printf "%-30s %s\n" "PPISP:" "✗ OFF"

    echo ""
    echo "📊 HYPERPARAMETERS"
    printf "%-30s %s\n" "Opacity Reg:" "$OPACITY_REG"
    printf "%-30s %s\n" "Scale Reg:" "$SCALE_REG"
    printf "%-30s %s\n" "Feature Dim:" "$FEATURE_DIM"
    printf "%-30s %s\n" "App Embed Dim:" "$APP_EMBED_DIM"
    [ "$ABSGRAD" = true ] && printf "%-30s %s\n" "Grow Grad2D:" "$GROW_GRAD2D"
    [ "$BILATERAL" = true ] && printf "%-30s %s\n" "Bilateral Grid Shape:" "($BILATERAL_SHAPE_X, $BILATERAL_SHAPE_Y, $BILATERAL_SHAPE_W)"
    [ "$PPISP" = true ] && printf "%-30s %s\n" "PPISP:" "✓ ON" || printf "%-30s %s\n" "PPISP:" "✗ OFF"

    echo ""
    echo "📂 PATHS"
    printf "%-30s %s\n" "Data Directory:" "$DATA_DIR"
    printf "%-30s %s\n" "COLMAP Directory:" "$COLMAP_DIR"
    printf "%-30s %s\n" "Result Directory:" "$RESULT_DIR"
    printf "%-30s %s\n" "Output Directory:" "$OUTPUT_DIR"
    
    echo ""
    echo "⏱️ TIMING"
    printf "%-30s %s\n" "Start Time:" "$(date '+%Y-%m-%d %H:%M:%S')"
    
} | tee "$LOG_FILE"

# Also save config in simple format
cp "$LOG_FILE" "$CONFIG_FILE"

print_info "Configuration logged to: $LOG_FILE"

# ============================================================================
# BUILD TRAINING FLAGS
# ============================================================================

FLAGS="--packed"
FLAGS="$FLAGS --disable_viewer"
FLAGS="$FLAGS --data_factor $DATA_FACTOR"
FLAGS="$FLAGS --opacity_reg $OPACITY_REG"
FLAGS="$FLAGS --scale_reg $SCALE_REG"
FLAGS="$FLAGS --feature_dim $FEATURE_DIM"
FLAGS="$FLAGS --max_steps $MAX_STEPS"

# Add conditional flags
if [ "$ABSGRAD" = true ]; then
    FLAGS="$FLAGS --strategy.absgrad --absgrad --strategy.grow_grad2d $GROW_GRAD2D"
fi

if [ "$ANTIALIASED" = true ]; then
    FLAGS="$FLAGS --antialiased"
fi

if [ "$APP_OPT" = true ]; then
    FLAGS="$FLAGS --app_opt --app_embed_dim $APP_EMBED_DIM"
fi

if [ "$BILATERAL" = true ]; then
    FLAGS="$FLAGS --post_processing bilateral_grid --bilateral_grid_fused"
    FLAGS="$FLAGS --bilateral_grid_shape $BILATERAL_SHAPE_X $BILATERAL_SHAPE_Y $BILATERAL_SHAPE_W"
fi

if [ "$PPISP" = true ]; then
    FLAGS="$FLAGS --post_processing ppisp"
fi

# ============================================================================
# SELECT DENSIFICATION STRATEGY
# ============================================================================

if [ "$DENSIFICATION" = "mcmc" ]; then
    DENSIFICATION_CMD="mcmc"
    INIT_OPA="0.5"
    INIT_SCALE="0.1"
else
    DENSIFICATION_CMD="default"
    INIT_OPA="0.1"
    INIT_SCALE="1.0"
fi

# ============================================================================
# CHECK GPU SETUP
# ============================================================================

print_header "GPU ENVIRONMENT CHECK"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate soccernet

srun python3 -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ Compute capability: {torch.cuda.get_device_capability()}')
"

# ============================================================================
# TRAINING
# ============================================================================

print_header "STARTING TRAINING"
print_info "Scene: $SCENE"
print_info "Densification: $DENSIFICATION"
print_info "Features: $FEATURE_NAME"
print_info "Output: $RESULT_DIR"

START_TIME=$(date +%s)

srun python "$REPO_ROOT/examples/simple_trainer.py" "$DENSIFICATION_CMD" \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --colmap_dir "$COLMAP_DIR" \
    --init_opa "$INIT_OPA" \
    --init_scale "$INIT_SCALE" \
    --no-normalize_world_space \
    --no-load_exposure \
    --test_every 0 \
    --save_steps 10000 20000 30000 40000 \
    $FLAGS \
    2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ============================================================================
# POST-TRAINING
# ============================================================================

print_header "TRAINING COMPLETED"
print_info "Duration: $((DURATION / 60)) minutes ($DURATION seconds)"
print_info "Results saved to: $RESULT_DIR"

# Get checkpoint
CKPT="$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1))_rank0.pt"

if [ -f "$CKPT" ]; then
    print_info "Checkpoint found: $(basename $CKPT)"
    
    # Run evaluation
    print_header "RUNNING EVALUATION"
    
    srun python "$REPO_ROOT/examples/eval_challenge.py" \
        --ckpt "$CKPT" \
        --data_dir "$CHALLENGE_DIR" \
        --result_folder "$(basename $OUTPUT_DIR)" \
        2>&1 | tee -a "$LOG_FILE"
else
    print_warning "Checkpoint not found at: $CKPT"
fi

# ============================================================================
# GENERATE SUMMARY
# ============================================================================

SUMMARY_FILE="$RESULT_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# Experiment Summary

## Configuration
- **Scene**: $SCENE
- **Densification**: $DENSIFICATION
- **Features**: $FEATURE_NAME
- **Run**: ${RUN_NUM}

## Parameters
- **Max Steps**: $MAX_STEPS
- **Data Factor**: $DATA_FACTOR
- **Opacity Reg**: $OPACITY_REG
- **Scale Reg**: $SCALE_REG

## Features Enabled
- Absgrad: $ABSGRAD
- Antialiased: $ANTIALIASED
- App Opt: $APP_OPT
- Bilateral Grid: $BILATERAL
- PPISP: $PPISP

## Results
- **Start Time**: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')
- **End Time**: $(date '+%Y-%m-%d %H:%M:%S')
- **Duration**: $((DURATION / 60)) minutes
- **Result Dir**: $RESULT_DIR
- **Checkpoint**: $CKPT

## Paths
- Data: $DATA_DIR
- COLMAP: $COLMAP_DIR
- Output: $OUTPUT_DIR
EOF

print_info "Summary saved to: $SUMMARY_FILE"

echo ""
print_header "✅ EXPERIMENT FINISHED SUCCESSFULLY"
