
import argparse
import os
import traceback

import torch
import torch.nn.functional as F
from torch import Tensor
import imageio
import numpy as np
from fused_ssim import fused_ssim
from typing import Any
from torch.utils.data import Dataset as DatasetTorch

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str,
                    help="Path to pretrained .pt checkpoint", default="/disk/SN-NVS-2026-raw/results-soccernet/scene-1-mcma_cfg-1-1/ckpts/ckpt_29999_rank0.pt")
parser.add_argument("--scene_dir",       default="/disk/SN-NVS-2026-raw/scene-1")
parser.add_argument("--colmap_dir",      default="/disk/SN-NVS-2026-raw/scene-1/sparse/0")
parser.add_argument("--ground_seg_dir",  default="/disk/SN-NVS-2026-raw/output_sam/grass_masks")
parser.add_argument("--data_factor",     type=int,   default=2)
parser.add_argument("--ground_lambda",   type=float, default=2.3)
parser.add_argument("--extra_steps",     type=int,   default=100,
                    help="Fine-tune steps after verification (0 = verify only)")
parser.add_argument("--lr",              type=float, default=1.6e-4)
parser.add_argument("--verify_only",     action="store_true")
parser.add_argument("--save_finetuned",  default="results/finetuned_ground.pt")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SEP  = "─" * 60

def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(msg, info=""): print(f"{PASS}  {msg}" + (f"  ({info})" if info else ""))
def fail(msg, info=""): print(f"{FAIL}  {msg}" + (f"  ({info})" if info else ""))
def sh(t): return f"shape={tuple(t.shape)} dtype={t.dtype}"
def save_debug_image(path, img):
    img = img.detach().cpu().numpy()

    # Fix shape
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    print(f"Save degub image {img.shape}")
    if img.shape[0] == 1:
        img = img.squeeze(0)
    # Normalize if float
    if img.dtype != np.uint8:
        img = img - img.min()
        if img.max() > 0:
            img = img / (img.max() + 1e-8)
        img = (img * 255).astype(np.uint8)

    imageio.imwrite(path, img)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — load checkpoint
# ═══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — load pretrained checkpoint")

ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
print(f"  checkpoint step : {ckpt.get('step', 'unknown')}")
print(f"  keys in ckpt    : {list(ckpt.keys())}")

splat_state = ckpt["splats"]
print(f"  splat params    : {list(splat_state.keys())}")

# Rebuild ParameterDict from saved state
splats = torch.nn.ParameterDict({
    k: torch.nn.Parameter(v.to(DEVICE))
    for k, v in splat_state.items()
})

N = splats["means"].shape[0]
ok("checkpoint loaded",    f"N={N} Gaussians  step={ckpt.get('step','?')}")
ok("means shape",          sh(splats["means"]))
ok("scales shape",         sh(splats["scales"]))
ok("quats shape",          sh(splats["quats"]))
ok("opacities shape",      sh(splats["opacities"]))
has_sh = "sh0" in splats and "shN" in splats
ok("SH coefficients",      f"sh0={sh(splats['sh0'])}") if has_sh else fail("SH missing")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — build dataset + grab one batch
# ═══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — dataset + one batch")
camtoworlds = None
Ks = None
W = H = 0

try:
    from datasets.colmap import Dataset, Parser

    par = Parser(
        data_dir   = args.scene_dir,
        factor     = args.data_factor,
        normalize  = False,
        test_every = 0,
        load_exposure = False,
        colmap_dir = args.colmap_dir,
    )

    ds = Dataset(
        par,
        split            = "train",
        patch_size       = None,
        load_depths      = True,
        load_ground_masks= args.ground_seg_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset=ds, batch_size=1, shuffle=False, num_workers=0
    )
    data = next(iter(loader))

    camtoworlds = data["camtoworld"].to(DEVICE)   # [1, 4, 4]
    print(f"Camtoworld matrix {camtoworlds}")
    Ks          = data["K"].to(DEVICE)            # [1, 3, 3]
    pixels      = data["image"].to(DEVICE) / 255.0
    H, W        = pixels.shape[1:3]

    ok("dataset loaded",       f"{len(ds)} training images")
    ok("batch image shape",    sh(pixels))

    has_mask = "ground_mask" in data
    has_pts  = "points" in data and "depths" in data

    ok("ground_mask present",  sh(data["ground_mask"])) if has_mask else fail("ground_mask missing — check ground_seg_dir")
    ok("colmap depth points",  f"M'={data['points'].shape[1]}") if has_pts else fail("points/depths missing")

    if has_mask:
        n_ground = data["ground_mask"].sum().item()
        ok("ground_mask has True pixels", f"{int(n_ground)} pixels") \
            if n_ground > 0 else fail("ground_mask all-False — SAM masks may be wrong files")

    DATASET_OK = True
except Exception as e:
    fail(f"Dataset failed: {e}")
    traceback.print_exc()
    DATASET_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — rasterize with rendered depth  (RGB+ED)
# ═══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — rasterize pretrained splats  (RGB+ED)")

depths_image = means2d = None
try:
    from gsplat.rendering import rasterization

    colors_in = torch.cat([splats["sh0"], splats["shN"]], dim=1) \
                if has_sh else splats.get("features", splats["sh0"])

    means = splats["means"]
    quats = F.normalize(splats["quats"], p=2, dim=-1)
    scales = torch.exp(splats["scales"])
    render_colors, render_alphas, info = rasterization(
        means       = means,
        quats       = quats,
        scales      = scales,
        opacities   = torch.sigmoid(splats["opacities"]),
        colors      = colors_in,
        viewmats    = torch.linalg.inv(camtoworlds),
        Ks          = Ks,
        width       = W,
        height      = H,
        sh_degree   = 3,
        near_plane  = 0.01,
        far_plane   = 1e10,
        render_mode = "RGB+ED",
        packed      = False,
    )

    ok("render shape",     sh(render_colors))
    ok("alpha shape",      sh(render_alphas))

    depths_image = render_colors[..., 3:4]   # [1, H, W, 1]
    means2d      = info["means2d"]            # [1, N, 2]

    # keep grad alive for backward
    if not means2d.requires_grad:
        means2d = means2d + 0.0 * splats["means"][..., None, :, :2].expand_as(means2d)

    d_min = depths_image[depths_image > 0].min().item() if (depths_image > 0).any() else 0
    d_max = depths_image.max().item()
    ok("rendered depth range", f"min={d_min:.3f}  max={d_max:.3f}")
    ok("means2d range",
       f"u∈[{means2d[...,0].min().item():.1f},{means2d[...,0].max().item():.1f}]  "
       f"v∈[{means2d[...,1].min().item():.1f},{means2d[...,1].max().item():.1f}]")
    render_colors_ = render_colors[...,:3]
    print(f"Shape Rendered Image {render_colors_.shape} Min {render_colors_.min()} Max {render_colors_.max()} ")

    save_debug_image("./logs/rgb_rendered.png", render_colors_)
    save_debug_image("./logs/depth_rendered.png", render_colors_)

    RASTER_OK = True
except Exception as e:
    fail(f"rasterization failed: {e}")
    traceback.print_exc()
    RASTER_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ground-plane depth prior  (D_gt)
# ═══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — ground_plane_depth_guided  (D_gt from masks)")

D_gt = None
try:
    from ground_plane_guided import ground_plane_depth_guided, quat_scale_to_covar

    assert DATASET_OK and has_mask, "Need dataset + ground_mask"

    ground_masks = data["ground_mask"].to(DEVICE)   # [1, H, W]

    covars = quat_scale_to_covar(
        F.normalize(splats["quats"], p=2, dim=-1),
        torch.exp(splats["scales"]),
    )
    save_debug_image("./logs/debug_mask.png", ground_masks.int() )
    _, _, _, _, D_gt, valid = ground_plane_depth_guided(
        means      = splats["means"],
        covars     = covars,
        viewmats   = torch.linalg.inv(camtoworlds),
        Ks         = Ks,
        masks      = ground_masks.unsqueeze(0),   # [1, 1, H, W]
        width      = W,
        height     = H,
        near_plane = 0.01,
        far_plane  = 1e10,
    )
    save_debug_image("./logs/D_t.png", D_gt)

    n_valid = int((D_gt > 0).sum().item())
    ok("D_gt shape",            sh(D_gt))
    ok("D_gt has ground depth", f"nonzero={n_valid}  "
       f"min={D_gt[D_gt>0].min().item():.3f}  max={D_gt.max().item():.3f}") \
        if n_valid > 0 else fail("D_gt all zeros — masks may not cover any ground")
    ok("valid Gaussians",       f"{int(valid.sum().item())}/{N}")

    GROUND_OK = True
except Exception as e:
    fail(f"ground_plane_depth_guided failed: {e}")
    traceback.print_exc()
    GROUND_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — depth supervision loss + backward
# ═══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — depth_supervision_loss + backward")

ground_loss = None
try:
    from ground_plane_guided import depth_supervision_loss

    assert RASTER_OK and GROUND_OK, "Need raster + D_gt"

    # sample rendered depth at each Gaussian's projected position
    norm_x  = (means2d[..., 0] / (W - 1)) * 2 - 1
    norm_y  = (means2d[..., 1] / (H - 1)) * 2 - 1
    grid_gd = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # [1, N, 1, 2]

    D_pred = F.grid_sample(
        depths_image.permute(0, 3, 1, 2),
        grid_gd,
        align_corners  = True,
        mode           = "bilinear",
        padding_mode   = "zeros",
    ).squeeze(1).squeeze(-1)   # [1, N]

    print(f"D_pred Points {D_pred[:3]}")
    ground_loss = depth_supervision_loss(
        D_pred  = D_pred,
        D_gt    = D_gt,
        means2d = means2d,
        width   = W,
        height  = H,
    )

    ok("ground_loss value",    f"{ground_loss.item():.6f}")
    ok("ground_loss finite",   torch.isfinite(ground_loss).item())

    # quick backward to confirm grad flow
    total = ground_loss * args.ground_lambda
    total.backward()
    print(f"Ground Loss {total}")
    grad = splats["means"].grad
    ok("means.grad exists",    grad is not None)
    ok("means.grad finite",
       grad is not None and torch.isfinite(grad).all().item())
    print(f"norm={grad.norm().item():.4f}" if grad is not None else "")

    LOSS_OK = True
except Exception as e:
    fail(f"depth_supervision_loss failed: {e}")
    traceback.print_exc()
    LOSS_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
section("VERIFICATION SUMMARY")

steps = {
    "1 - checkpoint loaded":           True,   # always reached if we got here
    "2 - dataset + ground masks":      DATASET_OK,
    "3 - rasterization RGB+ED":        RASTER_OK,
    "4 - ground_plane_depth_guided":   GROUND_OK,
    "5 - depth_supervision_loss":      LOSS_OK,
}
all_ok = all(v for v in steps.values())
for name, status in steps.items():
    print(f"{PASS if status else FAIL}  {name}")

if not all_ok:
    print("\n\033[91m  Pipeline has issues — fix above before fine-tuning.\033[0m")
    if not args.verify_only:
        raise SystemExit(1)

if args.verify_only or not all_ok:
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════════
# Fine-tune loop  (with GroundDiag)
# ══════════════════════════════════════════════════════════════════════════════

section(f"Fine-tune  ({args.extra_steps} steps  λ={args.ground_lambda})")
 
# ── diagnostic logger ─────────────────────────────────────────────────────────
from ground_diag import GroundDiag
diag = GroundDiag(diag_dir="./logs", log_every=10)
 
# ── optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam([
    {"params": [splats["means"]],     "lr": args.lr,        "name": "means"},
    {"params": [splats["scales"]],    "lr": 5e-3,           "name": "scales"},
    {"params": [splats["quats"]],     "lr": 1e-3,           "name": "quats"},
    {"params": [splats["opacities"]], "lr": 5e-2,           "name": "opacities"},
    {"params": [splats["sh0"]],       "lr": 2.5e-3,         "name": "sh0"},
    {"params": [splats["shN"]],       "lr": 2.5e-3 / 20,   "name": "shN"},
], eps=1e-15, betas=(0.9, 0.999))
 
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.01 ** (1.0 / max(args.extra_steps, 1))
)
 
loader_ft   = torch.utils.data.DataLoader(
    ds, batch_size=1, shuffle=True, num_workers=2, persistent_workers=True
)
loader_iter = iter(loader_ft)
 
import tqdm
pbar = tqdm.tqdm(range(args.extra_steps), desc="fine-tune")
 
for step in pbar:
    # ── fetch batch ──────────────────────────────────────────────────────────
    try:
        data_ft = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader_ft)
        data_ft     = next(loader_iter)
 
    c2w_ft = data_ft["camtoworld"].to(DEVICE)
    Ks_ft  = data_ft["K"].to(DEVICE)
    pix_ft = data_ft["image"].to(DEVICE) / 255.0
    H_ft, W_ft = pix_ft.shape[1:3]
    gm_ft  = data_ft["ground_mask"].to(DEVICE) if "ground_mask" in data_ft else None
 
    # ── forward render ───────────────────────────────────────────────────────
    colors_in = torch.cat([splats["sh0"], splats["shN"]], dim=1) # [1,N,K,3]
    
    rc, ra, info_ft = rasterization(
        means     = splats["means"],
        quats     = splats["quats"],
        scales    = torch.exp(splats["scales"]),
        opacities = torch.sigmoid(splats["opacities"]),
        colors    = colors_in,
        viewmats  = torch.linalg.inv(c2w_ft),
        Ks        = Ks_ft,
        width     = W_ft, height = H_ft,
        sh_degree = 3, near_plane = 0.01, far_plane = 1e10,
        render_mode = "RGB+ED", packed = False,
    )
    rgb_ft  = rc[..., :3] # RGB
    dep_ft  = rc[..., 3:4] # Depth
    m2d_ft  = info_ft["means2d"] # Means_2d Camera Coordinates
    if not m2d_ft.requires_grad:
        m2d_ft = m2d_ft + 0.0 * splats["means"][..., None, :, :2].expand_as(m2d_ft)
        # If the tensor is detached from the gradient history
        # Multiply by 0.0 splats["means"] tensor of zeros samee size that carries de gradient history
        # 
    # ── colour loss ──────────────────────────────────────────────────────────
    l1 = F.l1_loss(rgb_ft, pix_ft)

    ssim = 1.0 - fused_ssim(
        rgb_ft.permute(0, 3, 1, 2),
        pix_ft.permute(0, 3, 1, 2), padding="valid",
    )
    loss = l1 * 0.8 + ssim * 0.2

 
    # ── ground depth loss ────────────────────────────────────────────────────
    g_loss_val = 0.0
    D_gt_ft    = None
 
    if gm_ft is not None and gm_ft.sum() > 0:
        covars_ft = quat_scale_to_covar(
            F.normalize(splats["quats"], dim=-1),
            torch.exp(splats["scales"]),
        )
        _, _, _, _, D_gt_ft, valid_ft = ground_plane_depth_guided(
            means     = splats["means"],
            covars    = covars_ft,
            viewmats  = torch.linalg.inv(c2w_ft),
            Ks        = Ks_ft,
            masks     = gm_ft.unsqueeze(0),
            width     = W_ft, height = H_ft,
            near_plane = 0.01, far_plane = 1e10,
        )
        print(f"Shape D_gt_ft {D_gt_ft.shape}")
        print(f"Shape rgb_ft {rgb_ft.shape}")

        nx  = (m2d_ft[..., 0] / (W_ft - 1)) * 2 - 1
        ny  = (m2d_ft[..., 1] / (H_ft - 1)) * 2 - 1
        g   = torch.stack([nx, ny], dim=-1).unsqueeze(2)
        Dp  = F.grid_sample(
            dep_ft.permute(0, 3, 1, 2), g,
            align_corners=True, mode="bilinear", padding_mode="zeros",
        ).squeeze(1).squeeze(-1)
 
        g_loss = depth_supervision_loss(Dp, D_gt_ft, m2d_ft, W_ft, H_ft)
        if torch.isfinite(g_loss):
            loss        = loss + g_loss * args.ground_lambda
            g_loss_val  = g_loss.item()
        
    # ── backward ─────────────────────────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
 
    pbar.set_description(
        f"step={step}  loss={loss.item():.4f}  "
        f"l1={l1.item():.4f}  g={g_loss_val:.5f}"
    )
 
    # ── diagnostic step ───────────────────────────────────────────────────────
    diag.step(
        step        = step,
        loss        = loss.item(),
        l1          = l1.item(),
        ground_loss = g_loss_val,
        rgb_pred    = rgb_ft,
        rgb_gt      = pix_ft,
        depth_pred  = dep_ft,
        D_gt        = D_gt_ft,
        means2d     = m2d_ft,
        means_3d    = splats["means"],
        ground_mask = gm_ft,
    )
 