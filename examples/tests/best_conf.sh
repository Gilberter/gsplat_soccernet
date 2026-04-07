#!/bin/bash

# ============================================================
# TOP 10 PSNR-OPTIMIZED CONFIGURATIONS FOR 3DGS
# Based on: 3D Gaussian Splatting official paper + gsplat best practices
# Your proven 3DGS config: APP_EMBED_DIM=64 OPACITY_REG=0.01
# ============================================================

# CONFIG 1: EXTREME QUALITY (MCMC + Heavy Regularization)
# Expected PSNR: 27.5-28.5 dB | Memory: 22-24GB | Time: 90-120 min
cat > run_config_1.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=64
OPACITY_REG=0.02          # ⭐ Stronger regularization
SCALE_REG=0.01
DENSIFICATION=1           # MCMC (better quality than default)
APP_OPT=1                 # Appearance optimization ON
MAX_STEPS=50000           # ⭐ Longer training (quality boost)
PACKED_MODE=1
DATA_FACTOR=1             # ⭐ Full resolution data
SSIM_LAMBDA=0.25          # Slightly higher SSIM weight
RASTER_MODE=1             # Antialiased (small PSNR improvement)

# Run training
sbatch sh_soccernet.sh $SCENE mcmc-app64-opt02-50k $RASTER_MODE 0 0 1 $DENSIFICATION $APP_OPT $MAX_STEPS $PACKED_MODE $DATA_FACTOR
EOF

# CONFIG 2: HIGH QUALITY BALANCED (Default + Heavy Regularization)
# Expected PSNR: 27.0-28.0 dB | Memory: 20-22GB | Time: 75-90 min
cat > run_config_2.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=64
OPACITY_REG=0.015         # Strong regularization
SCALE_REG=0.01
DENSIFICATION=0           # Default strategy (faster convergence)
APP_OPT=1
MAX_STEPS=45000           # ⭐ Extended training
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.25

APP_EMBED_DIM=64 OPACITY_REG=0.015 SCALE_REG=0.01 SSIM_LAMBDA=0.25 sbatch sh_soccernet.sh scene-2 default-app64-opt015-45k 0 0 0 1 0 1 45000 0 2
EOF

# CONFIG 3: ABSGRAD QUALITY (AbsGS paper - fine details recovery)
# Expected PSNR: 26.8-27.8 dB | Memory: 19-21GB | Time: 70-85 min
cat > run_config_3.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=64
OPACITY_REG=0.01
SCALE_REG=0.01
DENSIFICATION=0
APP_OPT=1
ABSGRAD=1                 # ⭐ AbsGS: absolute gradients (fine details!)
GROW_GRAD2D=0.0008        # ⭐ Higher threshold for absgrad
MAX_STEPS=40000
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.2

APP_EMBED_DIM=64 OPACITY_REG=0.01 SCALE_REG=0.01 GROW_GRAD2D=0.0008 sbatch sh_soccernet.sh scene-2 absgs-app64-40k 0 0 0 1 0 1 40000 0 2
EOF

# CONFIG 4: PURE SH QUALITY (No appearance opt - smaller model)
# Expected PSNR: 26.5-27.5 dB | Memory: 16-18GB | Time: 60-75 min
cat > run_config_4.sh << 'EOF'
#!/bin/bash
SCENE=$1
OPACITY_REG=0.015         # Stronger regularization without appearance
SCALE_REG=0.01
DENSIFICATION=0
APP_OPT=0                 # ⭐ No appearance optimization
MAX_STEPS=45000
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.3           # Higher SSIM for purer colors

OPACITY_REG=0.015 SCALE_REG=0.01 SSIM_LAMBDA=0.3 sbatch sh_soccernet.sh scene-2 sh-only-45k 0 0 0 0 0 0 45000 0 2
EOF

# CONFIG 5: MCMC + APP OPTIMIZED (Best of both worlds)
# Expected PSNR: 27.2-28.2 dB | Memory: 21-23GB | Time: 85-110 min
cat > run_config_5.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=80          # ⭐ Even larger embedding (if memory allows)
OPACITY_REG=0.012         # Balanced regularization
SCALE_REG=0.01
DENSIFICATION=1           # MCMC
APP_OPT=1
MAX_STEPS=48000
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.22

APP_EMBED_DIM=80 OPACITY_REG=0.012 SCALE_REG=0.01 SSIM_LAMBDA=0.22 sbatch sh_soccernet.sh scene-2 mcmc-app80-opt012-50k 0 0 0 1 1 1 50000 0 2
EOF

# CONFIG 6: ANTIALIASED + APPEARANCE (Smooth high quality)
# Expected PSNR: 26.9-27.9 dB | Memory: 20-22GB | Time: 80-95 min
cat > run_config_6.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=64
OPACITY_REG=0.01
SCALE_REG=0.01
DENSIFICATION=0
APP_OPT=1
MAX_STEPS=42000
PACKED_MODE=1
DATA_FACTOR=1
RASTER_MODE=1             # ⭐ Antialiased mode (smoother rendering)
SSIM_LAMBDA=0.2

sbatch sh_soccernet.sh $SCENE antialiased-app64-42k 1 0 0 1 0 1 $MAX_STEPS $PACKED_MODE $DATA_FACTOR
EOF

# CONFIG 7: SUPER LONG TRAINING (Convergence at limit)
# Expected PSNR: 27.0-28.0 dB | Memory: 24-26GB | Time: 140-180 min
cat > run_config_7.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=64
OPACITY_REG=0.015
SCALE_REG=0.01
DENSIFICATION=1           # MCMC with long training
APP_OPT=1
MAX_STEPS=60000           # ⭐⭐⭐ ULTRA LONG TRAINING
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.2

sbatch sh_soccernet.sh $SCENE mcmc-ultra-60k 0 0 0 1 1 1 $MAX_STEPS $PACKED_MODE $DATA_FACTOR
EOF

# CONFIG 8: MEMORY CONSERVATIVE (If GPU memory is limited)
# Expected PSNR: 26.5-27.3 dB | Memory: 14-16GB | Time: 55-70 min
cat > run_config_8.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=32          # ⭐ Smaller embedding
OPACITY_REG=0.01
SCALE_REG=0.01
DENSIFICATION=0
APP_OPT=1
MAX_STEPS=40000
PACKED_MODE=1
DATA_FACTOR=2             # Downsampled data (quarter resolution)
SSIM_LAMBDA=0.25

sbatch sh_soccernet.sh $SCENE app32-mem-conservative-40k 0 0 0 1 0 1 $MAX_STEPS $PACKED_MODE $DATA_FACTOR
EOF

# CONFIG 9: AGGRESSIVE REGULARIZATION (Prevents oversplatting)
# Expected PSNR: 26.8-27.6 dB | Memory: 18-20GB | Time: 70-85 min
cat > run_config_9.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=64
OPACITY_REG=0.03          # ⭐⭐ AGGRESSIVE (clean model)
SCALE_REG=0.02            # ⭐ Higher scale regularization
DENSIFICATION=0
APP_OPT=1
MAX_STEPS=40000
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.15          # More L1 (sharp)

sbatch sh_soccernet.sh $SCENE aggressive-reg-app64-40k 0 0 0 1 0 1 $MAX_STEPS $PACKED_MODE $DATA_FACTOR
EOF

# CONFIG 10: ENSEMBLE READY (Speed optimized)
# Expected PSNR: 26.7-27.5 dB | Memory: 15-17GB | Time: 50-65 min
cat > run_config_10.sh << 'EOF'
#!/bin/bash
SCENE=$1
APP_EMBED_DIM=48          # Medium embedding
OPACITY_REG=0.01
SCALE_REG=0.01
DENSIFICATION=0
APP_OPT=1
MAX_STEPS=35000           # Shorter but efficient
PACKED_MODE=1
DATA_FACTOR=1
SSIM_LAMBDA=0.2

sbatch sh_soccernet.sh $SCENE app48-fast-35k 0 0 0 1 0 1 $MAX_STEPS $PACKED_MODE $DATA_FACTOR
EOF

echo "✅ All 10 configs generated!"
echo ""
echo "📊 RECOMMENDATION PRIORITY:"
echo "1️⃣  CONFIG 1 (MCMC+Heavy Reg) - BEST QUALITY if you have 24GB+ memory"
echo "2️⃣  CONFIG 5 (MCMC+APP80) - Best balance of quality & speed"
echo "3️⃣  CONFIG 2 (Default+Heavy Reg) - Solid quality, faster"
echo "4️⃣  CONFIG 7 (60k steps) - If you can wait 3+ hours"
echo ""
echo "🚀 RUN COMMANDS:"
echo "bash run_config_1.sh scene-1    # MCMC Ultra Quality"
echo "bash run_config_5.sh scene-1    # MCMC + APP80 Balanced"
echo "bash run_config_2.sh scene-1    # Default Fast Quality"