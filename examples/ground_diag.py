"""
ground_diag.py
==============
Drop-in diagnostic logger for the ground-plane depth fine-tune loop.
Call `diag.step(...)` every iteration and `diag.close()` at the end.

Outputs (all written to --diag_dir, default  ./diag/):
────────────────────────────────────────────────────────
Per-step (CSV, updated live):
  metrics.csv          — all scalar losses per step

Every log_every steps (images):
  rgb_gt_NNNNN.png     — ground-truth image
  rgb_pred_NNNNN.png   — rendered RGB
  depth_raw_NNNNN.png  — raw rendered depth (heat-map)
  depth_gt_NNNNN.png   — D_gt ground-plane depth prior (heat-map)
  depth_err_NNNNN.png  — |D_pred - D_gt| error map  (only at ground pixels)
  ground_overlay_NNNNN.png — RGB with ground mask + sparse COLMAP pts overlaid
  flatness_NNNNN.png   — per-Gaussian height above z=0 (scatter, 2-D top-view)

End of run (summaries):
  loss_curves.png      — loss / ground_loss / l1 over steps
  depth_improvement.png — D_gt vs D_pred scatter, first vs last logged step
  z0_histogram.png     — histogram of rendered depth at ground pixels,
                         first vs last step  (should sharpen around true value)
  summary.txt          — human-readable pass/fail report
"""

import os
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import imageio

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib import colormaps

def apply_float_colormap_np(img: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """
    NumPy version of apply_float_colormap

    Args:
        img: (..., 1) float32 in [0,1]
    Returns:
        (..., 3) float32 in [0,1]
    """
    print(f"Img Colormap {img.shape}")
    img = np.nan_to_num(img, nan=0.0)

    if colormap == "gray":
        return np.repeat(img, 3, axis=-1)

    img_long = (img * 255).astype(np.int32)

    if img_long.min() < 0 or img_long.max() > 255:
        raise ValueError(f"Values out of range: min={img_long.min()}, max={img_long.max()}")

    cmap = np.array(colormaps[colormap].colors)  # (256,3)
    print(f"Img Colormap {img.shape}")
    return cmap[img_long[..., 0]]  # map each pixel


def _depth_to_rgb(
    depth: np.ndarray,
    near_plane: float = None,
    far_plane: float = None,
    mask: np.ndarray = None

) -> np.ndarray:
    """
    NumPy version of apply_depth_colormap

    Args:
        depth: (..., 1) float32
        acc: (..., 1) optional mask
    Returns:
        (..., 3) float32 in [0,1]
    """
    depth = np.nan_to_num(depth, nan=0.0)
    #print(f"depth Colormap {depth.shape}")
    if depth.ndim == 2:
        depth = depth[..., None]  # (H, W) → (H, W, 1)
    if near_plane is None:
        near_plane = float(np.min(depth))
    if far_plane is None:
        far_plane = float(np.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = np.clip(depth, 0.0, 1.0)

    img = apply_float_colormap_np(depth, colormap="turbo")
    #print(f"depth Colormap {img.shape}")
    if mask is not None:
        mask = mask[..., None]  # (H,W) → (H,W,1)

        img = img * mask + (1.0 - mask)

    return img




def _overlay_points(img: np.ndarray, pts: np.ndarray,
                    color=(255, 220, 0), radius=3) -> np.ndarray:
    """Draw circles on img (H×W×3 uint8) at pixel positions pts (M×2)."""
    out = img.copy()
    H, W = img.shape[:2]
    for u, v in pts:
        u, v = int(round(u)), int(round(v))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    yy, xx = v + dy, u + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        out[yy, xx] = color
    return out


def _save(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
    imageio.imwrite(str(path), img)

def _turbo(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    cmap = colormaps["turbo"]
    return (cmap(x)[..., :3] * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic class
# ─────────────────────────────────────────────────────────────────────────────
class GroundDiag:
    """
    Usage inside the fine-tune loop:
    ─────────────────────────────────
        diag = GroundDiag(diag_dir="./diag", log_every=50)

        for step in range(extra_steps):
            ...
            diag.step(
                step        = step,
                loss        = loss.item(),
                l1          = l1.item(),
                ground_loss = g_loss_val,          # float, 0 if no ground pix
                rgb_pred    = rgb_ft,              # [1,H,W,3]  float32 [0,1]
                rgb_gt      = pix_ft,              # [1,H,W,3]  float32 [0,1]
                depth_pred  = dep_ft,              # [1,H,W,1]  float32
                D_gt        = D_gt_ft,             # [1,H,W]    float32 (or None)
                means2d     = m2d_ft,              # [1,N,2]    float32
                ground_mask = gm_ft,               # [1,H,W]    bool    (or None)
                means_3d    = splats["means"],     # [N,3]
            )

        diag.close()
    """

    def __init__(self, diag_dir: str = "./diag", log_every: int = 50):
        self.diag_dir  = Path(diag_dir)
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

        # scalar history
        self.history = {"step": [], "loss": [], "l1": [], "ground_loss": [],
                        "depth_mae": [], "depth_rmse": [],
                        "ground_pix_frac": [], "z0_mean": [], "z0_std": []}

        # CSV
        self._csv_path = self.diag_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=list(self.history.keys())
        )
        self._csv_writer.writeheader()

        # snapshots for first/last comparison
        self._first_snapshot = None
        self._last_snapshot  = None
        self._step_count     = 0

        print(f"  [GroundDiag] writing diagnostics → {self.diag_dir}/")

    # ────────────────────────────────────────────────────────────────────
    def step(self, step: int, loss: float, l1: float, ground_loss: float,
             rgb_pred: torch.Tensor, rgb_gt: torch.Tensor,
             depth_pred: torch.Tensor, D_gt: Optional[torch.Tensor],
             means2d: torch.Tensor, means_3d: torch.Tensor,
             ground_mask: Optional[torch.Tensor] = None):

        # ── detach everything ────────────────────────────────────────────
        def _np(t):
            if t is None: return None
            return t.detach().float().cpu().numpy()

        rp  = _np(rgb_pred)[0]           # H W 3
        rg  = _np(rgb_gt)[0]             # H W 3
        dp  = _np(depth_pred)[0, ..., 0] # H W
        dgt = _np(D_gt)[0]  if D_gt  is not None else None   # H W
        gm  = _np(ground_mask)[0].astype(bool) \
              if ground_mask is not None else np.ones(dp.shape, bool)
        m2d = _np(means2d)[0]            # N 2
        m3d = _np(means_3d)              # N 3

        H, W = dp.shape

        # ── per-pixel depth at ground mask ───────────────────────────────
        dp_ground = dp[gm] if gm.any() else np.array([])
        gt_ground = dgt[gm] if (dgt is not None and gm.any()) else np.array([])

        mae   = float(np.abs(dp_ground - gt_ground).mean()) \
                if (gt_ground.size > 0 and dp_ground.size > 0) else 0.0
        rmse  = float(np.sqrt(((dp_ground - gt_ground)**2).mean())) \
                if (gt_ground.size > 0 and dp_ground.size > 0) else 0.0
        z_mean = float(dp_ground.mean()) if dp_ground.size > 0 else 0.0
        z_std  = float(dp_ground.std())  if dp_ground.size > 0 else 0.0
        gpf    = float(gm.mean())

        # ── record scalars ───────────────────────────────────────────────
        row = dict(step=step, loss=loss, l1=l1, ground_loss=ground_loss,
                   depth_mae=mae, depth_rmse=rmse,
                   ground_pix_frac=gpf, z0_mean=z_mean, z0_std=z_std)
        for k, v in row.items():
            self.history[k].append(v)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # ── store snapshot for comparison ────────────────────────────────
        snap = dict(dp_ground=dp_ground, gt_ground=gt_ground,
                    dp=dp, dgt=dgt, gm=gm, step=step)
        if self._first_snapshot is None:
            self._first_snapshot = snap
        self._last_snapshot = snap
        

        # ── periodic image saves ─────────────────────────────────────────
        if self._step_count % self.log_every == 0:
            tag = f"{step:05d}"

            # 1. RGB comparison
            side = np.concatenate([
                (np.clip(rg, 0, 1) * 255).astype(np.uint8),
                (np.clip(rp, 0, 1) * 255).astype(np.uint8),
            ], axis=1)
            _save(self.diag_dir / f"rgb_compare_{tag}.png", side)

            # 2. Raw rendered depth heat-map
            vmin = dp[dp > 0].min() if (dp > 0).any() else 0
            vmax = dp.max()
            _save(self.diag_dir / f"depth_raw_{tag}.png",
                  _depth_to_rgb(dp))

            # 3. D_gt heat-map
            if dgt is not None:
                vmin_g = dgt[dgt > 0].min() if (dgt > 0).any() else 0
                vmax_g = dgt.max()
                _save(self.diag_dir / f"depth_gt_{tag}.png",
                      _depth_to_rgb(dgt, vmin_g, vmax_g, mask=(dgt > 0)))

                # 4. Absolute error map (only where D_gt > 0)
                valid = (dgt > 0)
                err   = np.zeros_like(dp)
                err[valid] = np.abs(dp[valid] - dgt[valid])
                e_max = err[valid].max() if valid.any() else 1.0
                _save(self.diag_dir / f"depth_err_{tag}.png",
                      _depth_to_rgb(err, 0, e_max, mask=valid))

            # 5. Ground overlay: RGB + mask contour + COLMAP pts
            rgb_u8 = (np.clip(rg, 0, 1) * 255).astype(np.uint8).copy()
            # tint ground pixels with a semi-transparent green
            rgb_u8[gm] = (rgb_u8[gm] * 0.5 +
                          np.array([0, 200, 80], np.float32) * 0.5).astype(np.uint8)
            # overlay in-image COLMAP points
            in_img = ((m2d[:, 0] >= 0) & (m2d[:, 0] < W) &
                      (m2d[:, 1] >= 0) & (m2d[:, 1] < H))
            rgb_u8 = _overlay_points(rgb_u8, m2d[in_img], color=(255, 220, 0))
            _save(self.diag_dir / f"ground_overlay_{tag}.png", rgb_u8)

            # 6. Depth side-by-side with error bar
            if dgt is not None and valid.any():
                vmin_c = min(vmin, dgt[dgt > 0].min())
                vmax_c = max(vmax, dgt.max())
                d_pred_rgb = _depth_to_rgb(dp,  vmin_c, vmax_c)
                d_gt_rgb   = _depth_to_rgb(dgt, vmin_c, vmax_c, mask=(dgt > 0))
                err_rgb    = _depth_to_rgb(err, 0, e_max, mask=valid)
                trio = np.concatenate([d_gt_rgb, d_pred_rgb, err_rgb], axis=1)
                _save(self.diag_dir / f"depth_trio_{tag}.png", trio)

            # 7. Flatness view: top-down scatter of Gaussian Z-positions
            self._save_flatness(m3d, tag)

            # 8. Ground-pixel depth histogram
            if dp_ground.size > 0 and gt_ground.size > 0:
                self._save_hist(dp_ground, gt_ground, tag)

        self._step_count += 1

    # ────────────────────────────────────────────────────────────────────
    def _save_flatness(self, m3d: np.ndarray, tag: str):
        """Top-down view: X-Y plane, colour = Z (height above ground)."""
        x, y, z = m3d[:, 0], m3d[:, 1], m3d[:, 2]
        # raster into a 256×256 image
        sz  = 256
        x_n = (x - x.min()) / (x.ptp() + 1e-8)
        y_n = (y - y.min()) / (y.ptp() + 1e-8)
        z_n = (z - z.min()) / (z.ptp() + 1e-8)

        canvas = np.zeros((sz, sz, 3), np.uint8)
        px = np.clip((x_n * (sz - 1)).astype(int), 0, sz - 1)
        py = np.clip((y_n * (sz - 1)).astype(int), 0, sz - 1)
      
        cols = _turbo(z_n)  # (N, 3)

        canvas[py, px] = cols

        # draw z=0 reference line (min-z row)
        z0_row = int(((0 - z.min()) / (z.ptp() + 1e-8)) * (sz - 1))
        z0_row = np.clip(z0_row, 0, sz - 1)
        canvas[z0_row, :] = [0, 255, 120]
        
        _save(self.diag_dir / f"flatness_{tag}.png", (canvas))

    def _save_hist(self, dp_g: np.ndarray, gt_g: np.ndarray, tag: str):
        """Simple ASCII-art depth histogram saved as a small PNG."""
        bins = 32
        lo   = min(dp_g.min(), gt_g.min())
        hi   = max(dp_g.max(), gt_g.max()) + 1e-8
        edges = np.linspace(lo, hi, bins + 1)
        c_pred, _ = np.histogram(dp_g, bins=edges)
        c_gt,   _ = np.histogram(gt_g, bins=edges)

        # render to a small uint8 image (2 rows of bars)
        h_img, w_img = 80, bins * 8
        img = np.ones((h_img, w_img, 3), np.uint8) * 30  # dark bg

        for i, (cp, cg) in enumerate(zip(c_pred, c_gt)):
            max_h = max(c_pred.max(), c_gt.max(), 1)
            hp = int(cp / max_h * (h_img // 2 - 4))
            hg = int(cg / max_h * (h_img // 2 - 4))
            x0, x1 = i * 8, i * 8 + 7
            # D_pred = blue
            img[h_img // 2 - hp: h_img // 2, x0:x1] = [80, 140, 255]
            # D_gt   = green
            img[h_img // 2 + 2: h_img // 2 + 2 + hg, x0:x1] = [80, 230, 120]

        _save(self.diag_dir / f"depth_hist_{tag}.png", img)

    # ────────────────────────────────────────────────────────────────────
    def close(self):
        """Write summary plots and close CSV."""
        self._csv_file.close()

        steps = np.array(self.history["step"])
        if len(steps) == 0:
            print("  [GroundDiag] no steps recorded — skipping summary.")
            return

        # ── loss curves PNG ──────────────────────────────────────────────
        self._plot_loss_curves(steps)

        # ── D_gt vs D_pred scatter (first vs last) ───────────────────────
        self._plot_depth_scatter()

        # ── z0 histogram (first vs last) ────────────────────────────────
        self._plot_z0_hist()

        # ── summary.txt ─────────────────────────────────────────────────
        self._write_summary(steps)

        print(f"  [GroundDiag] diagnostics saved → {self.diag_dir}/")

    # ────────────────────────────────────────────────────────────────────
    def _plot_loss_curves(self, steps):
        """Minimal ASCII-chart saved as a PNG."""
        keys    = ["loss", "l1", "ground_loss", "depth_mae"]
        colors  = [(255,180,60),(100,180,255),(100,230,120),(255,100,100)]
        h, w    = 200, 640
        img     = np.ones((h, w, 3), np.uint8) * 20
        n       = len(steps)

        for ki, (key, col) in enumerate(zip(keys, colors)):
            vals = np.array(self.history[key], dtype=float)
            if vals.max() == vals.min(): continue
            v_n  = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            xs   = (np.arange(n) / max(n - 1, 1) * (w - 1)).astype(int)
            ys   = (h - 4 - v_n * (h - 8)).astype(int)
            for i in range(len(xs) - 1):
                x0, y0, x1, y1 = xs[i], ys[i], xs[i+1], ys[i+1]
                # Bresenham line
                dx, dy = abs(x1-x0), abs(y1-y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy
                while True:
                    if 0 <= x0 < w and 0 <= y0 < h:
                        img[y0, x0] = col
                    if x0 == x1 and y0 == y1: break
                    e2 = 2 * err
                    if e2 > -dy: err -= dy; x0 += sx
                    if e2 <  dx: err += dx; y0 += sy

        # legend strip at bottom
        legend_y = h - 12
        for ki, (key, col) in enumerate(zip(keys, colors)):
            lx = ki * 160 + 4
            img[legend_y:legend_y+8, lx:lx+20] = col
        _save(self.diag_dir / "loss_curves.png", img)

    def _plot_depth_scatter(self):
        """2D scatter D_gt (x) vs D_pred (y) — first and last snapshot."""
        if self._first_snapshot is None: return
        snaps = [self._first_snapshot, self._last_snapshot]
        sz    = 128
        out   = []
        for sn in snaps:
            img  = np.ones((sz, sz, 3), np.uint8) * 25
            dp_g = sn["dp_ground"]
            gt_g = sn["gt_ground"]
            if dp_g.size == 0: out.append(img); continue
            lo   = min(dp_g.min(), gt_g.min())
            hi   = max(dp_g.max(), gt_g.max()) + 1e-8
            xn   = ((gt_g - lo) / (hi - lo) * (sz - 2)).astype(int)
            yn   = ((dp_g - lo) / (hi - lo) * (sz - 2)).astype(int)
            # diagonal (perfect prediction line)
            for i in range(sz):
                img[sz - 1 - i, i] = [80, 80, 80]
            for xi, yi in zip(np.clip(xn,0,sz-1), np.clip(yn,0,sz-1)):
                img[sz - 1 - yi, xi] = [100, 210, 255]
            out.append(img)
        combined = np.concatenate(out, axis=1)
        _save(self.diag_dir / "depth_improvement.png", combined)

    def _plot_z0_hist(self):
        """Histogram of D_pred at ground pixels — first vs last."""
        if self._first_snapshot is None: return
        snaps = [self._first_snapshot, self._last_snapshot]
        bins  = 48
        h_img, w_img = 80, bins * 8
        combined = []
        for sn in snaps:
            img  = np.ones((h_img, w_img, 3), np.uint8) * 30
            dp_g = sn["dp_ground"]
            gt_g = sn["gt_ground"]
            if dp_g.size == 0:
                combined.append(img); continue
            lo  = min(dp_g.min(), gt_g.min() if gt_g.size > 0 else dp_g.min())
            hi  = max(dp_g.max(), gt_g.max() if gt_g.size > 0 else dp_g.max()) + 1e-8
            edges = np.linspace(lo, hi, bins + 1)
            cp, _ = np.histogram(dp_g, bins=edges)
            cg, _ = np.histogram(gt_g, bins=edges) if gt_g.size > 0 else (cp * 0, None)
            mx = max(cp.max(), cg.max(), 1)
            for i, (vp, vg) in enumerate(zip(cp, cg)):
                hp = int(vp / mx * (h_img // 2 - 4))
                hg = int(vg / mx * (h_img // 2 - 4))
                x0, x1 = i * 8, i * 8 + 7
                img[h_img//2 - hp: h_img//2, x0:x1] = [80, 140, 255]
                img[h_img//2 + 2: h_img//2 + 2 + hg, x0:x1] = [80, 230, 120]
            combined.append(img)
        _save(self.diag_dir / "z0_histogram.png",
              np.concatenate(combined, axis=0))

    def _write_summary(self, steps):
        n = len(steps)
        h = self.history

        # improvements
        def delta(key):
            v = h[key]
            if len(v) < 2: return 0.0
            return v[-1] - v[0]

        lines = [
            "=" * 60,
            "  GROUND-DEPTH FINE-TUNE DIAGNOSTIC SUMMARY",
            "=" * 60,
            f"  Steps logged       : {n}",
            f"  Final total loss   : {h['loss'][-1]:.5f}",
            f"  Δ total loss       : {delta('loss'):+.5f}",
            f"  Final l1           : {h['l1'][-1]:.5f}",
            f"  Δ l1               : {delta('l1'):+.5f}",
            f"  Final ground_loss  : {h['ground_loss'][-1]:.6f}",
            f"  Δ ground_loss      : {delta('ground_loss'):+.6f}",
            "",
            "  DEPTH QUALITY AT GROUND PIXELS",
            f"  Final depth MAE    : {h['depth_mae'][-1]:.4f}",
            f"  Δ depth MAE        : {delta('depth_mae'):+.4f}",
            f"  Final depth RMSE   : {h['depth_rmse'][-1]:.4f}",
            f"  Δ depth RMSE       : {delta('depth_rmse'):+.4f}",
            "",
            "  Z=0 PLANE ADHERENCE (rendered depth at ground pix)",
            f"  Initial z_mean     : {h['z0_mean'][0]:.4f}",
            f"  Final   z_mean     : {h['z0_mean'][-1]:.4f}",
            f"  Initial z_std      : {h['z0_std'][0]:.4f}",
            f"  Final   z_std      : {h['z0_std'][-1]:.4f}",
            "",
            "  PASS/FAIL",
        ]

        checks = {
            "ground_loss decreasing"     : delta("ground_loss") < 0,
            "depth_mae improving"        : delta("depth_mae")   < 0,
            "z_std decreasing (flatter)" : delta("z0_std")      < 0,
            "total loss not exploding"   : h["loss"][-1] < h["loss"][0] * 3,
        }
        for name, ok in checks.items():
            lines.append(f"  {'  PASS' if ok else '  FAIL'}  {name}")

        lines += ["", "  See metrics.csv for full per-step data.", "=" * 60]

        summary_path = self.diag_dir / "summary.txt"
        summary_path.write_text("\n".join(lines))
        print("\n".join(lines))