# debug_depth_loss.py
"""
Quick diagnostic for the DA3 depth loss in training.
Run this BEFORE training to verify the depth prior pipeline is correct.
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

from datasets.colmap import Parser, Dataset
from utils_eval_challenge import load_splats, render_camera

data_dir   = "/disk/SN-NVS-2026-raw/scene-1"
colmap_dir = os.path.join(data_dir, "sparse/0")
depth_dir  = os.path.join(data_dir, "scene-1-dae3/depth_maps.npz")
ckpt_dir   = f"{data_dir}-challenge/results-cfg_2-3-2-1/ckpt_29999_rank0.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ok(msg, info=""): print(f"{PASS}  {msg}" + (f"  ({info})" if info else ""))
def fail(msg, info=""): print(f"{FAIL}  {msg}" + (f"  ({info})" if info else ""))

# ─── 1. Dataset ──────────────────────────────────────────────────────────────
parser = Parser(
    data_dir=data_dir, factor=2, normalize=False,
    test_every=0, load_exposure=False, colmap_dir=colmap_dir,
)
trainset = Dataset(parser, split="train", load_depths=True, load_mini_npz=depth_dir)
loader   = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

data = next(iter(loader))
print("\n=== STEP 1: Data shapes ===")
for k, v in data.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:20s}: {tuple(v.shape)}  dtype={v.dtype}  "
              f"min={v.float().min():.3f}  max={v.float().max():.3f}")

# ─── 2. Depth prior shape & values ───────────────────────────────────────────
print("\n=== STEP 2: mini_depth analysis ===")
mini_depth_raw = data["mini_depth"]   # what the dataloader gives you
print(f"  raw shape from loader : {tuple(mini_depth_raw.shape)}")
# Expected usable shape: [1, H, W] or [1, 1, H, W]

# What the training code does:
mini_depth_permuted = mini_depth_raw.permute(0, 2, 3, 1)
print(f"  after permute(0,2,3,1): {tuple(mini_depth_permuted.shape)}")
# This should be [1, H_small, W_small, 1] to match depth_rendered [1, H, W, 1]

H_img, W_img = data["image"].shape[1:3]
H_d,   W_d   = mini_depth_permuted.shape[1:3]
print(f"  image resolution      : H={H_img}  W={W_img}")
print(f"  depth resolution      : H={H_d}   W={W_d}")

if H_d != H_img or W_d != W_img:
    fail("RESOLUTION MISMATCH — depth prior and rendered depth have different sizes!")
    print(f"  → The valid mask will be nearly all-False because shapes don't broadcast.")
    print(f"  → Fix: upsample mini_depth to image resolution before comparison.")
else:
    ok("Resolutions match")

# ─── 3. Simulate what training does with the alignment ───────────────────────
print("\n=== STEP 3: Simulate training depth loss ===")

means, quats, scales, opacities, colors, sh_degree, app_module, features = \
    load_splats([ckpt_dir], DEVICE)

c2w = data["camtoworld"][0].cpu().numpy()
K   = data["K"][0].cpu().numpy()

raw_render, _ = render_camera(
    means, quats, scales, opacities, colors, sh_degree,
    c2w, K, W_img, H_img,
    near=0.01, far=1e10, device=DEVICE,
    app_module=app_module, features=features,
)

depth_rendered_np = raw_render[0, ..., 3]                   # [H, W]
depth_rendered    = torch.from_numpy(depth_rendered_np).to(DEVICE).unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]

print(f"  depth_rendered shape  : {tuple(depth_rendered.shape)}")
print(f"  depth_rendered range  : {depth_rendered.min():.3f} .. {depth_rendered.max():.3f}")
print(f"  depth_rendered >0 px  : {(depth_rendered > 0).sum().item():,}")

# ── What training does (permute makes it [1, H_small, W_small, 1]) ───────────
depth_prior_training = mini_depth_raw.to(DEVICE).permute(0, 2, 3, 1)  # [1, H_small, W_small, 1]
print(f"\n  depth_prior (training code): {tuple(depth_prior_training.shape)}")
print(f"  depth_prior range           : {depth_prior_training.min():.3f} .. {depth_prior_training.max():.3f}")
print(f"  depth_prior >0 px           : {(depth_prior_training > 0).sum().item():,}")

valid_training = (depth_rendered > 0) & (depth_prior_training > 0)
print(f"\n  valid mask (training logic) : {valid_training.sum().item():,} pixels")

if valid_training.sum() == 0:
    fail("ZERO VALID PIXELS — depth loss is computing 0 every step!")
    print("  Likely cause: shape mismatch means the & gives all-False")
else:
    ok(f"valid pixels: {valid_training.sum().item():,}")

# ── Correct approach: upsample prior to image resolution ─────────────────────
print("\n=== STEP 4: Correct approach (upsample prior) ===")

depth_prior_up = F.interpolate(
    mini_depth_raw.to(DEVICE),          # [1, 1, H_small, W_small]
    size=(H_img, W_img),
    mode="bilinear",
    align_corners=False,
).squeeze(1).unsqueeze(-1)              # [1, H, W, 1]

print(f"  depth_prior upsampled : {tuple(depth_prior_up.shape)}")
valid_correct = (depth_rendered > 0) & (depth_prior_up > 0)
print(f"  valid pixels          : {valid_correct.sum().item():,}")

if valid_correct.sum() > 100:
    r = depth_rendered[valid_correct].detach()
    p = depth_prior_up[valid_correct].detach()

    r_mean = r.mean();  p_mean = p.mean()
    r_var  = ((r - r_mean)**2).mean().clamp(min=1e-8)
    s = ((r - r_mean) * (p - p_mean)).mean() / r_var
    t = p_mean - s * r_mean

    print(f"  alignment scale s     : {s:.4f}")
    print(f"  alignment shift t     : {t:.4f}")

    depth_aligned = s * depth_rendered + t
    loss = F.l1_loss(depth_aligned[valid_correct], depth_prior_up[valid_correct])
    print(f"  depth L1 loss         : {loss.item():.6f}")

    if loss.item() < 1e-6:
        fail("Loss is essentially zero — check scale alignment or depth values")
    elif loss.item() > 100:
        fail(f"Loss is very large ({loss.item():.1f}) — units may be wrong")
    else:
        ok(f"Depth loss looks reasonable: {loss.item():.4f}")

# ─── 5. Save visual diagnostic ───────────────────────────────────────────────
print("\n=== STEP 5: Saving diagnostic images ===")
os.makedirs("debug_depth_loss", exist_ok=True)

def to_colormap(x_np, cmap=cv2.COLORMAP_JET):
    x = x_np.copy().astype(np.float32)
    x = np.nan_to_num(x)
    mask = x > 0
    if mask.sum() == 0:
        return np.zeros((*x.shape, 3), dtype=np.uint8)
    xmin, xmax = x[mask].min(), x[mask].max()
    x_norm = np.zeros_like(x)
    x_norm[mask] = (x[mask] - xmin) / (xmax - xmin + 1e-8)
    return cv2.applyColorMap((x_norm * 255).astype(np.uint8), cmap)

# RGB GT
rgb_gt = cv2.cvtColor(data["image"][0].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

# Rendered depth
dr_vis = to_colormap(depth_rendered_np)

# Prior at small res (what training uses — WRONG if shapes mismatch)
dp_small_vis = to_colormap(
    cv2.resize(depth_prior_training[0, ..., 0].cpu().numpy(),
               (W_img, H_img), interpolation=cv2.INTER_NEAREST)
)

# Prior upsampled (CORRECT)
dp_up_vis = to_colormap(depth_prior_up[0, ..., 0].cpu().numpy())

# valid mask comparison
valid_train_vis = (valid_training[0, ..., 0].cpu().numpy() * 255).astype(np.uint8)
valid_train_vis = cv2.cvtColor(valid_train_vis, cv2.COLOR_GRAY2BGR)

valid_correct_vis = (valid_correct[0, ..., 0].cpu().numpy() * 255).astype(np.uint8)
valid_correct_vis = cv2.cvtColor(valid_correct_vis, cv2.COLOR_GRAY2BGR)

row1 = np.concatenate([rgb_gt,          dr_vis],         axis=1)
row2 = np.concatenate([dp_small_vis,    dp_up_vis],      axis=1)
row3 = np.concatenate([valid_train_vis, valid_correct_vis], axis=1)

# Add text labels
def label(img, text):
    return cv2.putText(img.copy(), text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

row1 = np.concatenate([label(rgb_gt,       "GT RGB"),
                       label(dr_vis,        "Rendered depth")], axis=1)
row2 = np.concatenate([label(dp_small_vis, "DA3 depth (small, WRONG size)"),
                       label(dp_up_vis,     "DA3 depth (upsampled, CORRECT)")], axis=1)
row3 = np.concatenate([label(valid_train_vis,   "Valid mask WRONG"),
                       label(valid_correct_vis,  "Valid mask CORRECT")], axis=1)

canvas = np.concatenate([row1, row2, row3], axis=0)
cv2.imwrite("debug_depth_loss/diagnostic.png", canvas)
ok("Saved → debug_depth_loss/diagnostic.png")

print("\n=== SUMMARY ===")
print(f"  mini_depth shape from loader : {tuple(mini_depth_raw.shape)}")
print(f"  image resolution             : {H_img}x{W_img}")
print(f"  depth resolution             : {H_d}x{W_d}")
print(f"  shapes match?                : {H_d == H_img and W_d == W_img}")
print(f"  valid px (training code)     : {valid_training.sum().item():,}")
print(f"  valid px (after upsample)    : {valid_correct.sum().item():,}")
print()
print("FIX for trainer_scene_prior.py:")
print("  Replace:")
print("    depth_prior = depth_prior.permute(0, 2, 3, 1)")
print("  With:")
print("    depth_prior = F.interpolate(")
print("        depth_prior,                    # [1, 1, H_small, W_small]")
print("        size=(height, width),")
print("        mode='bilinear',")
print("        align_corners=False,")
print("    ).permute(0, 2, 3, 1)              # [1, H, W, 1]")