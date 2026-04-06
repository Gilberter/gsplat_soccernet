#!/bin/bash
#SBATCH --job-name=gs_experiment
#SBATCH --output=./logs/experiment_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_experiment_%j.log
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --mem=25G
#SBATCH --time=04:00:00

# =============================================================================
# ORGANIZED GAUSSIAN SPLATTING EXPERIMENT RUNNER
# =============================================================================
#
# FOLDER STRUCTURE PRODUCED:
#
#   results-soccernet/
#     <scene>/
#       densification/
#         absgrad-default/
#         mcmc/
#         default/
#       app_opt/
#         on/
#         off/
#       embed_dim/          ← app_opt auto-enabled
#         dim8/
#         dim16/
#         dim32/
#         dim64/
#       opacity_reg/
#         reg0001/
#         reg001/
#         reg005/
#         reg01/
#       scale_reg/
#         reg0001/
#         reg001/
#         reg005/
#         reg01/
#       raster_mode/
#         classic/
#         antialiased/
#       packed/
#         on/
#         off/
#       full/
#         <your-label>/     ← free-form, set flags via env vars
#
# USAGE:
#   sbatch run_experiment.sh <scene> <experiment_group> <variant>
#
# EXAMPLES:
#   sbatch run_experiment.sh scene-2 densification mcmc
#   sbatch run_experiment.sh scene-2 densification absgrad-default
#   sbatch run_experiment.sh scene-2 app_opt on
#   sbatch run_experiment.sh scene-2 embed_dim dim32
#   sbatch run_experiment.sh scene-2 opacity_reg reg005
#   sbatch run_experiment.sh scene-2 scale_reg reg001
#   sbatch run_experiment.sh scene-2 raster_mode antialiased
#   sbatch run_experiment.sh scene-2 packed off
#   sbatch run_experiment.sh scene-2 full my-custom-combo   # use env vars
#
# OVERRIDE ANY HYPERPARAMETER via env vars before submitting:
#   MAX_STEPS=30000 OPACITY_REG=0.05 sbatch run_experiment.sh scene-2 opacity_reg reg005
#
# =============================================================================

mkdir -p ./logs

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
print(f'Arch list: {torch.cuda.get_arch_list()}')
"

# =============================================================================
# POSITIONAL ARGUMENTS
# =============================================================================
SCENE=${1:?"ERROR: Provide scene name. Usage: sbatch run_experiment.sh <scene> <group> <variant>"}
EXPERIMENT_GROUP=${2:?"ERROR: Provide experiment group: densification|app_opt|embed_dim|opacity_reg|scale_reg|raster_mode|packed|full"}
VARIANT=${3:?"ERROR: Provide variant (e.g. mcmc, on, dim32, reg001, baseline)"}

# =============================================================================
# GLOBAL DEFAULTS  (override via env vars)
# =============================================================================
MAX_STEPS=${MAX_STEPS:-40000}
DATA_FACTOR=${DATA_FACTOR:-2}
SH_DEGREE=${SH_DEGREE:-3}

# Regularisation
OPACITY_REG=${OPACITY_REG:-0.01}
SCALE_REG=${SCALE_REG:-0.01}
SSIM_LAMBDA=${SSIM_LAMBDA:-0.2}

# Architecture
APP_EMBED_DIM=${APP_EMBED_DIM:-16}
FEATURE_DIM=${FEATURE_DIM:-32}

# Densification
GROW_GRAD2D=${GROW_GRAD2D:-0.0008}

# Feature flags  (0=OFF  1=ON)
APP_OPT=${APP_OPT:-0}
ABSGRAD=${ABSGRAD:-1}
PACKED_MODE=${PACKED_MODE:-1}
ANTIALIASED=${ANTIALIASED:-0}
BILATERAL=${BILATERAL:-0}
MCMC=${MCMC:-0}

# Bilateral grid dims
BILATERAL_SHAPE_X=${BILATERAL_SHAPE_X:-16}
BILATERAL_SHAPE_Y=${BILATERAL_SHAPE_Y:-16}
BILATERAL_SHAPE_W=${BILATERAL_SHAPE_W:-8}

# =============================================================================
# EXPERIMENT GROUP → AUTO-CONFIGURE RELEVANT HYPERPARAMETERS
# =============================================================================

case "$EXPERIMENT_GROUP" in

  densification)
    case "$VARIANT" in
      mcmc)            MCMC=1; ABSGRAD=0 ;;
      absgrad-default) MCMC=0; ABSGRAD=1 ;;
      default)         MCMC=0; ABSGRAD=0 ;;
      *) echo "❌ Unknown densification variant: $VARIANT (valid: mcmc | absgrad-default | default)"; exit 1 ;;
    esac
    ;;

  app_opt)
    case "$VARIANT" in
      on)  APP_OPT=1 ;;
      off) APP_OPT=0 ;;
      *) echo "❌ Unknown app_opt variant: $VARIANT (valid: on | off)"; exit 1 ;;
    esac
    ;;

  embed_dim)
    APP_OPT=1   # embed_dim experiments require app_opt enabled
    case "$VARIANT" in
      dim8)  APP_EMBED_DIM=8  ;;
      dim16) APP_EMBED_DIM=16 ;;
      dim32) APP_EMBED_DIM=32 ;;
      dim64) APP_EMBED_DIM=64 ;;
      *) echo "❌ Unknown embed_dim variant: $VARIANT (valid: dim8 | dim16 | dim32 | dim64)"; exit 1 ;;
    esac
    ;;

  opacity_reg)
    case "$VARIANT" in
      reg0001) OPACITY_REG=0.001 ;;
      reg001)  OPACITY_REG=0.01  ;;
      reg005)  OPACITY_REG=0.05  ;;
      reg01)   OPACITY_REG=0.1   ;;
      *) echo "❌ Unknown opacity_reg variant: $VARIANT (valid: reg0001 | reg001 | reg005 | reg01)"; exit 1 ;;
    esac
    ;;

  scale_reg)
    case "$VARIANT" in
      reg0001) SCALE_REG=0.001 ;;
      reg001)  SCALE_REG=0.01  ;;
      reg005)  SCALE_REG=0.05  ;;
      reg01)   SCALE_REG=0.1   ;;
      *) echo "❌ Unknown scale_reg variant: $VARIANT (valid: reg0001 | reg001 | reg005 | reg01)"; exit 1 ;;
    esac
    ;;

  raster_mode)
    case "$VARIANT" in
      classic)     ANTIALIASED=0 ;;
      antialiased) ANTIALIASED=1 ;;
      *) echo "❌ Unknown raster_mode variant: $VARIANT (valid: classic | antialiased)"; exit 1 ;;
    esac
    ;;

  packed)
    case "$VARIANT" in
      on)  PACKED_MODE=1 ;;
      off) PACKED_MODE=0 ;;
      *) echo "❌ Unknown packed variant: $VARIANT (valid: on | off)"; exit 1 ;;
    esac
    ;;

  full)
    # Free-form: VARIANT is just a label. Configure everything via env vars.
    echo "ℹ️  Free-form experiment '$VARIANT' — all flags from env vars."
    ;;

  *)
    echo "❌ Unknown experiment group: $EXPERIMENT_GROUP"
    echo "   Valid: densification | app_opt | embed_dim | opacity_reg | scale_reg | raster_mode | packed | full"
    exit 1
    ;;
esac

# =============================================================================
# DERIVED FLAGS + LABELS
# =============================================================================

if   [ "$MCMC"    -eq 1 ]; then
    DENSIFICATION_STRATEGY="mcmc"
    DENSIFICATION_LABEL="mcmc"
elif [ "$ABSGRAD" -eq 1 ]; then
    DENSIFICATION_STRATEGY="default --strategy.absgrad --absgrad --strategy.grow_grad2d $GROW_GRAD2D"
    DENSIFICATION_LABEL="absgrad-default"
else
    DENSIFICATION_STRATEGY="default --strategy.grow_grad2d $GROW_GRAD2D"
    DENSIFICATION_LABEL="default"
fi

RASTERIZE_MODE=$([ "$ANTIALIASED" -eq 1 ] && echo "antialiased" || echo "classic")

FLAGS=""
[ "$BILATERAL"   -eq 1 ] && FLAGS="$FLAGS --bilateral_grid_fused --post_processing bilateral_grid \
                                           --bilateral_grid_shape $BILATERAL_SHAPE_X $BILATERAL_SHAPE_Y $BILATERAL_SHAPE_W"
[ "$ANTIALIASED" -eq 1 ] && FLAGS="$FLAGS --antialiased"
[ "$APP_OPT"     -eq 1 ] && FLAGS="$FLAGS --app_opt --app_embed_dim $APP_EMBED_DIM"
[ "$PACKED_MODE" -eq 1 ] && FLAGS="$FLAGS --packed"

# =============================================================================
# OUTPUT PATHS — organised hierarchy
# =============================================================================

BASE=/disk/SN-NVS-2026-raw
DATA_DIR=$BASE/${SCENE}
CHALLENGE_DIR=$BASE/${SCENE}-challenge
COLMAP_DIR=$DATA_DIR/sparse/0

# Primary result path:  results-soccernet/<scene>/<group>/<variant>/
HIER_BASE=$BASE/results-soccernet/${SCENE}/${EXPERIMENT_GROUP}/${VARIANT}
RESULT_DIR=$HIER_BASE

# Auto-increment run index if the folder already exists
if [ -d "$RESULT_DIR" ]; then
    i=1
    while [ -d "${HIER_BASE}_run${i}" ]; do ((i++)); done
    RESULT_DIR="${HIER_BASE}_run${i}"
    echo "⚠️  Output folder exists → using $RESULT_DIR"
fi

CKPT=$RESULT_DIR/ckpts/ckpt_$((MAX_STEPS-1))_rank0.pt

# Challenge evaluation output mirrors the same hierarchy
OUTPUT_DIR=$CHALLENGE_DIR/${EXPERIMENT_GROUP}/${VARIANT}
mkdir -p "$OUTPUT_DIR" "$RESULT_DIR/ckpts" ./logs

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               EXPERIMENT CONFIGURATION                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-25s %-35s ║\n" "Scene:"             "$SCENE"
printf "║  %-25s %-35s ║\n" "Experiment group:"  "$EXPERIMENT_GROUP"
printf "║  %-25s %-35s ║\n" "Variant:"           "$VARIANT"
printf "║  %-25s %-35s ║\n" "Max steps:"         "$MAX_STEPS"
printf "║  %-25s %-35s ║\n" "Data factor:"       "$DATA_FACTOR"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-25s %-35s ║\n" "Densification:"     "$DENSIFICATION_LABEL"
printf "║  %-25s %-35s ║\n" "Raster mode:"       "$RASTERIZE_MODE"
printf "║  %-25s %-35s ║\n" "Packed:"            "$([ "$PACKED_MODE" -eq 1 ] && echo ON || echo OFF)"
printf "║  %-25s %-35s ║\n" "App opt:"           "$([ "$APP_OPT"     -eq 1 ] && echo ON || echo OFF)"
printf "║  %-25s %-35s ║\n" "App embed dim:"     "$APP_EMBED_DIM"
printf "║  %-25s %-35s ║\n" "Bilateral:"         "$([ "$BILATERAL"   -eq 1 ] && echo ON || echo OFF)"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-25s %-35s ║\n" "Opacity reg:"       "$OPACITY_REG"
printf "║  %-25s %-35s ║\n" "Scale reg:"         "$SCALE_REG"
printf "║  %-25s %-35s ║\n" "SSIM lambda:"       "$SSIM_LAMBDA"
printf "║  %-25s %-35s ║\n" "Grow grad2d:"       "$GROW_GRAD2D"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-25s %-35s ║\n" "Result dir:"        "...${RESULT_DIR: -40}"
printf "║  %-25s %-35s ║\n" "Checkpoint:"        "...${CKPT: -40}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# TRAINING
# =============================================================================

echo "🏋️  Starting training at $(date)"
START_TIME=$(date +%s)

srun python /home/hensemberk/dev/Soccernet/gsplat/examples/simple_trainer.py $DENSIFICATION_STRATEGY \
    --max_steps $MAX_STEPS \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --data_factor $DATA_FACTOR \
    --no-normalize_world_space \
    --save_steps 10000 20000 30000 $MAX_STEPS \
    --no-load_exposure \
    --test_every 0 \
    --colmap_dir $COLMAP_DIR \
    --rasterize_mode $RASTERIZE_MODE \
    --disable_viewer \
    --sh_degree $SH_DEGREE \
    --opacity_reg $OPACITY_REG \
    --scale_reg $SCALE_REG \
    --batch_size 1 \
    $FLAGS

TRAIN_EXIT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅  Training finished | exit=$TRAIN_EXIT | ${DURATION}s (~$((DURATION/60)) min) | $(date)"
echo ""

# =============================================================================
# EVALUATION
# =============================================================================

srun python /home/hensemberk/dev/Soccernet/gsplat/examples/eval_challenge.py \
    --ckpt "$CKPT" \
    --data_dir "$CHALLENGE_DIR" \
    --result_folder "$(basename $OUTPUT_DIR)"

# =============================================================================
# SAVE REPRODUCIBLE RUN CONFIG
# =============================================================================

LOG_FILE=$RESULT_DIR/run_config.txt
mkdir -p "$(dirname "$LOG_FILE")"

cat <<EOL > "$LOG_FILE"
================================================================================
SLURM JOB INFO
================================================================================
Job ID:                $SLURM_JOB_ID
Node:                  $SLURM_NODELIST
User:                  $(whoami)
Start Time:            $(date -d @$START_TIME 2>/dev/null || date -r $START_TIME)
End Time:              $(date)
Duration (sec):        $DURATION

================================================================================
EXPERIMENT IDENTITY
================================================================================
Scene:                 $SCENE
Experiment Group:      $EXPERIMENT_GROUP
Variant:               $VARIANT

================================================================================
ENVIRONMENT
================================================================================
Conda Env:             $CONDA_DEFAULT_ENV
Python:                $(which python)
CUDA Visible Devices:  ${CUDA_VISIBLE_DEVICES:-"(not set)"}

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
Max Steps:             $MAX_STEPS
Data Factor:           $DATA_FACTOR
SH Degree:             $SH_DEGREE

================================================================================
DENSIFICATION
================================================================================
Strategy:              $DENSIFICATION_LABEL
Grow Grad2D:           $GROW_GRAD2D

================================================================================
TRAINING FLAGS
================================================================================
Raster Mode:           $RASTERIZE_MODE
Packed Mode:           $([ "$PACKED_MODE" -eq 1 ] && echo ON || echo OFF)
AbsGrad:               $([ "$ABSGRAD"     -eq 1 ] && echo ON || echo OFF)
App Opt:               $([ "$APP_OPT"     -eq 1 ] && echo ON || echo OFF)
Bilateral Grid:        $([ "$BILATERAL"   -eq 1 ] && echo ON || echo OFF)
Antialiased:           $([ "$ANTIALIASED" -eq 1 ] && echo ON || echo OFF)

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
FULL REPRODUCIBLE COMMAND
================================================================================
SCENE=$SCENE \\
MAX_STEPS=$MAX_STEPS DATA_FACTOR=$DATA_FACTOR \\
APP_OPT=$APP_OPT APP_EMBED_DIM=$APP_EMBED_DIM \\
OPACITY_REG=$OPACITY_REG SCALE_REG=$SCALE_REG \\
PACKED_MODE=$PACKED_MODE ANTIALIASED=$ANTIALIASED \\
BILATERAL=$BILATERAL ABSGRAD=$ABSGRAD MCMC=$MCMC \\
sbatch run_experiment.sh $SCENE $EXPERIMENT_GROUP $VARIANT

================================================================================
EOL

echo "📄  Run config saved → $LOG_FILE"
echo "✅  All done."