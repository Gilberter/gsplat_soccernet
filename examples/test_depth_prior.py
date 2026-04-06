import os
import torch
import cv2
import numpy as np

PASS = "\033[92m  PASS\033[0m"

from datasets.colmap import Parser, Dataset
from utils_eval_challenge import load_splats, render_camera

data_dir    = "/disk/SN-NVS-2026-raw/scene-1"
colmap_dir  = os.path.join(data_dir, "sparse/0")
depth_dir   = os.path.join(data_dir, "scene-1-dae3/depth_maps.npz")
ckpt_dir    = f"{data_dir}-challenge/results-cfg_2-3-2-1/ckpt_29999_rank0.pt"

parser = Parser(
    data_dir=data_dir,
    factor=2,
    normalize=False,
    test_every=0,
    load_exposure=False,
    colmap_dir=colmap_dir,
)
trainset = Dataset(
    parser,
    split="train",
    load_depths=True,
    load_mini_npz=depth_dir,
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True,
    num_workers=4, persistent_workers=True, pin_memory=True,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ok(msg, info=""): print(f"{PASS}  {msg}" + (f"  ({info})" if info else ""))
def sh(t): return f"shape={tuple(t.shape)} dtype={t.dtype} min={t.min():.3f} max={t.max():.3f}"

def colormap_depth(depth_np):
    """Normalize a float depth map and apply JET colormap. Returns BGR uint8."""
    d = depth_np.copy().astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    mask = d > 0
    if mask.sum() == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    dmin, dmax = d[mask].min(), d[mask].max()
    d_norm = np.zeros_like(d)
    d_norm[mask] = (d[mask] - dmin) / (dmax - dmin + 1e-8)
    d_u8 = (d_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)

def colormap_error(err_np, vmax=None):
    """Visualize an error map with INFERNO colormap."""
    e = err_np.copy().astype(np.float32)
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    mask = e > 0
    if vmax is None:
        vmax = e[mask].max() if mask.sum() > 0 else 1.0
    e_norm = np.clip(e / (vmax + 1e-8), 0, 1)
    e_u8 = (e_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(e_u8, cv2.COLORMAP_INFERNO)

means, quats, scales, opacities, colors, sh_degree, app_module, features = \
    load_splats([ckpt_dir], DEVICE)

os.makedirs("debug_depth", exist_ok=True)

for i, data in enumerate(trainloader):
    if i >= 5:
        break

    c2w    = data["camtoworld"][0].cpu().numpy()   # [4, 4]
    K      = data["K"][0].cpu().numpy()            # [3, 3]
    rgb_gt = data["image"][0].cpu().numpy()        # [H, W, 3] uint8

    # DA3 depth: [1, 1, H_small, W_small] → [H_small, W_small]
    depth_prior_small = data["mini_depth"][0, 0].cpu().numpy()

    H, W = rgb_gt.shape[:2]
    ok(f"[{i}] image", f"H={H} W={W}")
    ok(f"[{i}] depth_prior (raw)", sh(data["mini_depth"][0, 0]))

    # ── Render ──────────────────────────────────────────────────────────
    raw_render, n_rendered = render_camera(
        means, quats, scales, opacities, colors, sh_degree,
        c2w, K, W, H,
        near=0.01, far=1e10,
        device=DEVICE,
        app_module=app_module,
        features=features,
    )
    # raw_render: [1, H, W, 4]  — RGB in [0,1], depth in camera meters
    rgb_render  = raw_render[0, ..., :3]          # [H, W, 3]  float [0,1]
    depth_render = raw_render[0, ..., 3]          # [H, W]     float meters

    ok(f"[{i}] depth_render", f"min={depth_render.min():.3f} max={depth_render.max():.3f} mean={depth_render[depth_render>0].mean():.3f}")

    # ── Upsample DA3 depth to render resolution ──────────────────────────
    depth_prior = cv2.resize(
        depth_prior_small,
        (W, H),
        interpolation=cv2.INTER_LINEAR,
    )
    ok(f"[{i}] depth_prior (resized)", f"min={depth_prior.min():.3f} max={depth_prior.max():.3f}")

    # ── Align rendered depth scale to DA3 (least-squares s,t) ───────────
    valid = (depth_render > 0) & (depth_prior > 0)
    if valid.sum() > 100:
        r = depth_render[valid]
        p = depth_prior[valid]
        A = np.vstack([r, np.ones_like(r)]).T
        s, t = np.linalg.lstsq(A, p, rcond=None)[0]
        depth_render_aligned = s * depth_render + t
        ok(f"[{i}] alignment", f"scale={s:.4f}  shift={t:.4f}")
    else:
        depth_render_aligned = depth_render
        ok(f"[{i}] alignment", "skipped (too few valid pixels)")

    # ── Errors ────────────────────────────────────────────────────────────
    abs_diff = np.zeros_like(depth_render)
    rel_diff = np.zeros_like(depth_render)
    signed   = np.zeros_like(depth_render)

    if valid.sum() > 0:
        r_al = depth_render_aligned[valid]
        p_v  = depth_prior[valid]
        abs_diff[valid] = np.abs(r_al - p_v)
        rel_diff[valid] = np.abs(r_al - p_v) / (p_v + 1e-6)
        signed[valid]   = r_al - p_v

        mae  = abs_diff[valid].mean()
        rmse = np.sqrt((abs_diff[valid] ** 2).mean())
        print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  rel_MAE={rel_diff[valid].mean():.4f}")

    # ── Visualize ─────────────────────────────────────────────────────────
    # RGB: gt and render side by side  (convert to BGR for cv2)
    rgb_gt_u8     = rgb_gt.astype(np.uint8)                          # already uint8
    rgb_render_u8 = (np.clip(rgb_render, 0, 1) * 255).astype(np.uint8)

    rgb_gt_bgr     = cv2.cvtColor(rgb_gt_u8,     cv2.COLOR_RGB2BGR)
    rgb_render_bgr = cv2.cvtColor(rgb_render_u8, cv2.COLOR_RGB2BGR)

    depth_r_vis = colormap_depth(depth_render_aligned)
    depth_p_vis = colormap_depth(depth_prior)
    abs_vis     = colormap_error(abs_diff)
    rel_vis     = colormap_error(rel_diff, vmax=1.0)  # cap relative error at 100%

    signed_norm = (signed - signed.min()) / (signed.max() - signed.min() + 1e-8)
    signed_vis  = cv2.applyColorMap(
        (signed_norm * 255).astype(np.uint8), cv2.COLORMAP_COOL
    )

    # Build canvas: row1=RGB, row2=depth, row3=errors
    top = np.concatenate([rgb_gt_bgr,  rgb_render_bgr], axis=1)
    mid = np.concatenate([depth_r_vis, depth_p_vis],    axis=1)
    bot = np.concatenate([abs_vis,     rel_vis],         axis=1)
    canvas = np.concatenate([top, mid, bot], axis=0)

    out_path = f"debug_depth/compare_{i:03d}.png"
    cv2.imwrite(out_path, canvas)
    print(f"  Saved → {out_path}  ({n_rendered:,} Gaussians rendered)")