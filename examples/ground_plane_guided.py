"""
ground_plane_losses.py
======================
Three-level depth supervision strategy for soccer field Gaussian Splatting.

WHY THREE LEVELS:
─────────────────
  Level 1 — DA3 dense metric depth (direct image comparison)
             Supervises EVERY pixel. Gradient flows through the full
             alpha-blended rendered depth map.  Corrects the whole pitch.

  Level 2 — Geometric ground-plane prior (Zw=0 plane intersection)
             Supervises only SAM-masked GROUND pixels.  Pure geometry —
             no learned component, no scale ambiguity.  Enforces the flat
             Zw=0 constraint on floor Gaussians specifically.

  Level 3 — 2DGS normal consistency (from simple_trainer_2dgs.py)
             Forces floor disk normals to align with the depth gradient.
             On a flat floor the depth gradient is zero → disks must point
             straight up → they LIE IN the Zw=0 plane.
             This is the mechanism that prevents green Gaussians drifting
             in front of players.

WHAT WAS WRONG WITH THE ORIGINAL depth_supervision_loss:
─────────────────────────────────────────────────────────
  The original function receives D_pred with shape [..., C, N] where N
  is the number of Gaussians — NOT a pixel image.  It uses means2d to
  sample D_gt at Gaussian centres and compares the per-Gaussian camera-
  space z-depth (Zc) against the geometric prior sampled at the same pixel.

  Problems:
    1. Zc is the Gaussian's own camera-space z — it ignores all the
       alpha-blending that the rasterizer does.  A Gaussian at Zc=5.0
       may render to a pixel at depth 4.6 because closer Gaussians
       partially occlude it.  The loss is comparing the wrong quantity.
    2. It has no gradient signal for Gaussians that project to pixels
       where D_gt=0 (non-ground pixels), even if those Gaussians float
       above the floor.
    3. The correct rendered depth — what we actually see — is in
       depths_image [C, H, W, 1], the alpha-normalized expected depth
       that comes out of rasterization with render_mode="RGB+ED".

  The fix: compare depths_image directly against D_gt_map (same shape,
  both [C, H, W]).  Gradient flows through the rasterizer's backward
  pass and is correctly distributed over all contributing Gaussians.

ALPHA-BLENDED DEPTH — WHY IT'S BETTER:
────────────────────────────────────────
  The 2DGS paper (eq. 18) defines:

       z_mean = Σᵢ(ωᵢ · zᵢ) / (Σᵢ(ωᵢ) + ε)

  where ωᵢ = Tᵢ · αᵢ · Ĝᵢ(u(x)) is the blending weight of the i-th
  Gaussian at pixel x, and Tᵢ is the accumulated transmittance.

  This is exactly what gsplat returns in the last channel of render_colors
  when render_mode="RGB+ED" (Expected Depth).  The gradient of L1/MSE
  on this quantity with respect to means flows back to every Gaussian
  that has non-zero weight at any supervised pixel — correctly pulling
  each floor Gaussian toward the right depth proportional to its weight.

  The per-Gaussian Zc approach only moves one Gaussian at a time and
  ignores the collective rendering.

HOW THE GREEN-GAUSSIAN PROBLEM IS SOLVED:
──────────────────────────────────────────
  Green Gaussians appear in front of players because:
    (a) The color loss sees green grass through transparent players.
    (b) The optimizer places a green Gaussian at depth z_player to
        explain the residual green tint — it's cheaper than making
        the player Gaussian perfectly opaque.

  The three-level depth supervision prevents this:
    (a) Level 1 (DA3): the rendered depth at every player pixel must
        match the DA3 depth of the player surface.  A green Gaussian
        floating at player depth would make the rendered depth too
        shallow — the L1 loss pulls it back to the correct depth.
    (b) Level 2 (Zw=0 prior): only activates on SAM-masked ground pixels,
        so it does not directly constrain player Gaussians.  But it
        keeps floor Gaussians firmly at z_floor, which means the optimizer
        cannot place a floor Gaussian above z_floor to explain a green
        pixel at player depth.
    (c) Level 3 (normal consistency): floor disks are forced flat.  A
        disk lying in Zw=0 cannot have a component pointing toward the
        camera, so it cannot "see" any player pixel.  Its colour signal
        comes only from floor-facing pixels.

USAGE:
──────
  In simple_trainer_guided.py replace the ground_loss block with:

      from ground_plane_losses import (
          depth_from_da3_loss,
          ground_plane_prior_loss,
          build_ground_depth_map,
      )

  See each function's docstring for the exact call signature and where
  in the training loop to place it.
"""

import math
from typing import Optional, Tuple
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# LEVEL 1 — DA3 DENSE METRIC DEPTH
# =============================================================================

def depth_from_da3_loss(
    depths_image: Tensor,     # [C, H, W, 1]  rendered expected depth (RGB+ED)
    da3_depth:    Tensor,     # [C, H, W]     DA3 metric depth (already scale-aligned)
    ground_mask:  Optional[Tensor] = None,  # [C, H, W] bool — if given, weight floor 2×
    scene_scale:  float = 1.0,
    floor_weight: float = 2.0,
) -> Tensor:
    """
    Direct pixel-space L1 loss between the rasterizer's alpha-blended
    expected depth and the DA3 dense metric depth map.

    WHY THIS IS BETTER THAN per-Gaussian Zc comparison:
      - depths_image is the alpha-weighted rendered depth at each pixel.
        It is what the camera actually "sees".
      - Gradient flows through the full rasterizer backward pass, correctly
        weighting each Gaussian by its contribution (blending weight) to
        the pixel.  A Gaussian that barely contributes to a pixel gets a
        small gradient; the dominant Gaussian gets a large one.
      - L1 is more robust than MSE for depth — outliers from sky pixels
        or DA3 failures are down-weighted relative to MSE.

    WHERE TO CALL:
      After rasterization, before loss.backward().
      depths_image comes from renders[..., 3:4] when render_mode="RGB+ED".

    SCALE ALIGNMENT:
      da3_depth must be aligned to COLMAP metric scale before this call.
      Use align_depth_scale() from ground_plane_guided.py, passing the
      COLMAP sparse depth points as D_sparse.

    Args:
        depths_image : [C, H, W, 1]  — last channel of render_colors (RGB+ED)
        da3_depth    : [C, H, W]     — DA3 metric depth, scale-aligned
        ground_mask  : [C, H, W] bool — optional, upweights floor pixels
        scene_scale  : overall scene scale from parser.scene_scale * 1.1
        floor_weight : multiply loss at ground pixels by this factor (default 2×)

    Returns:
        Scalar loss tensor with gradient.
    """
    # depths_image: [C, H, W, 1] → [C, H, W]
    d_pred = depths_image[..., 0]           # [C, H, W]
    d_gt   = da3_depth.to(d_pred.device)    # [C, H, W]

    # Only supervise where DA3 has valid (positive) depth
    valid  = d_gt > 0.0                     # [C, H, W]
    if not valid.any():
        return torch.tensor(0.0, device=d_pred.device, requires_grad=True)

    diff = (d_pred[valid] - d_gt[valid]).abs()

    if ground_mask is not None:
        # Build per-pixel weights: floor pixels get floor_weight, rest get 1.0
        weights = torch.ones_like(d_pred)
        weights[ground_mask & valid] = floor_weight
        loss = (diff * weights[valid]).mean()
    else:
        loss = diff.mean()

    return loss * scene_scale


# =============================================================================
# LEVEL 2 — GEOMETRIC GROUND-PLANE PRIOR  (image-space version)
# =============================================================================

def build_ground_depth_map(
    viewmats:   Tensor,    # [C, 4, 4]  world-to-camera
    Ks:         Tensor,    # [C, 3, 3]  intrinsics
    masks:      Tensor,    # [C, H, W]  bool ground mask
    width:      int,
    height:     int,
    near_plane: float = 0.01,
    far_plane:  float = 1e10,
) -> Tensor:
    """
    Build D_gt as a full [C, H, W] depth image in pixel space.

    This replaces the per-Gaussian sampling approach.  The returned map
    has the geometric ground-truth depth at every ground pixel and 0
    everywhere else.  It can then be compared directly against
    depths_image using ground_plane_prior_loss().

    The math is unchanged from the original ground_plane_depth_guided():
      1. For each masked pixel (u, v) shoot a ray (Xc, Yc) = ((u-cx)/fx, (v-cy)/fy)
      2. Solve A·[Xw, Yw]ᵀ = b to find where the ray hits Zw=0  (eq. 7)
      3. Depth = Euclidean distance from camera centre to that world point

    Returns:
        D_gt : [C, H, W]  float32,  0 where not ground
    """
    C  = viewmats.shape[0]
    fx = Ks[:, 0, 0];  fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2];  cy = Ks[:, 1, 2]

    D_gt = torch.zeros(C, height, width, device=viewmats.device, dtype=viewmats.dtype)

    for c in range(C):
        mask_c = masks[c] if masks.dim() == 3 else masks[0, c]   # [H, W]
        ys, xs = torch.where(mask_c)
        if xs.numel() == 0:
            continue

        # Pinhole un-projection  (paper eq. 8)
        Xc = (xs.float() - cx[c]) / fx[c]   # [M]
        Yc = (ys.float() - cy[c]) / fy[c]   # [M]

        R  = viewmats[c, :3, :3]             # [3, 3]
        t  = viewmats[c, :3, 3]              # [3]

        # 2×2 system:  A · [Xw, Yw]ᵀ = b   (derived by substituting Zw=0)
        # A = [[Xc·r31 - r11,  Xc·r32 - r12],
        #      [Yc·r31 - r21,  Yc·r32 - r22]]
        # b = [t1 - Xc·t3,  t2 - Yc·t3]
        A11 = Xc * R[2, 0] - R[0, 0]        # [M]
        A12 = Xc * R[2, 1] - R[0, 1]        # [M]
        A21 = Yc * R[2, 0] - R[1, 0]        # [M]
        A22 = Yc * R[2, 1] - R[1, 1]        # [M]

        b1  = t[0] - Xc * t[2]              # [M]
        b2  = t[1] - Yc * t[2]              # [M]

        det = A11 * A22 - A12 * A21          # [M]

        # Skip degenerate pixels (camera parallel to ground)
        nonzero = det.abs() > 1e-8
        if not nonzero.any():
            continue

        # Cramer's rule
        Xw = torch.zeros_like(Xc)
        Yw = torch.zeros_like(Yc)
        Xw[nonzero] = (b1[nonzero] * A22[nonzero] - b2[nonzero] * A12[nonzero]) / det[nonzero]
        Yw[nonzero] = (A11[nonzero] * b2[nonzero] - A21[nonzero] * b1[nonzero]) / det[nonzero]

        # Camera centre in world coords:  c_world = -Rᵀ · t
        cam_centre = -(R.T @ t)              # [3]

        # Euclidean distance from camera centre to (Xw, Yw, 0)
        dx    = Xw - cam_centre[0]
        dy    = Yw - cam_centre[1]
        dz    = -cam_centre[2]               # Zw = 0  →  dz = 0 - cam_centre[2]
        depth = torch.sqrt(dx**2 + dy**2 + dz**2)   # [M]

        valid = (
            nonzero &
            (depth > near_plane) &
            (depth < far_plane)
        )

        if valid.any():
            D_gt[c, ys[valid], xs[valid]] = depth[valid]

    return D_gt   # [C, H, W]


def ground_plane_prior_loss(
    depths_image: Tensor,    # [C, H, W, 1]  rendered expected depth
    D_gt:         Tensor,    # [C, H, W]     from build_ground_depth_map()
    loss_type:    str = "l1",   # "l1" | "mse" | "huber"
    huber_delta:  float = 0.5,
) -> Tensor:
    """
    Image-space ground-plane depth supervision.

    Compares the rendered depth image against the geometric Zw=0 depth
    map at all ground pixels where D_gt > 0.

    WHY IMAGE-SPACE INSTEAD OF per-Gaussian:
    ─────────────────────────────────────────
    The old depth_supervision_loss samples D_gt at means2d positions and
    compares against Zc (the Gaussian's own z-depth, not the blended pixel
    depth).  This has two problems:

      Problem A — Wrong quantity:
        Zc is the Gaussian primitive's camera-space z.  The rendered depth
        at a pixel is Σ(ωᵢ·Zc_i) / Σ(ωᵢ), a weighted average.  A single
        Gaussian's Zc can be far from the pixel's rendered depth if other
        Gaussians dominate.  Comparing Zc against D_gt[pixel] conflates
        the primitive's position with the pixel's appearance.

      Problem B — Sparse gradient:
        The signal only reaches Gaussians whose means2d falls on a ground
        pixel.  Gaussians that are slightly off the floor but whose splatted
        footprint covers ground pixels get no signal.

    The image-space comparison fixes both:
      - depths_image IS the alpha-weighted rendered depth — the right quantity.
      - Every pixel in the ground mask contributes a gradient.  The rasterizer
        backward pass distributes it to every Gaussian with non-zero alpha at
        that pixel, weighted by its contribution.

    Args:
        depths_image : [C, H, W, 1]  rendered depth (last channel of RGB+ED)
        D_gt         : [C, H, W]     geometric ground-truth from build_ground_depth_map
        loss_type    : "l1" (recommended), "mse", or "huber"
        huber_delta  : threshold for huber loss (in metres)

    Returns:
        Scalar loss, or 0.0 with grad if no ground pixels are visible.
    """
    d_pred = depths_image[..., 0]        # [C, H, W]
    valid  = D_gt > 0.0                  # [C, H, W]

    if not valid.any():
        return torch.tensor(0.0, device=depths_image.device, requires_grad=True)

    pred_v = d_pred[valid]
    gt_v   = D_gt[valid]

    if loss_type == "l1":
        return F.l1_loss(pred_v, gt_v)
    elif loss_type == "mse":
        return F.mse_loss(pred_v, gt_v)
    elif loss_type == "huber":
        return F.huber_loss(pred_v, gt_v, delta=huber_delta)
    else:
        raise ValueError(f"loss_type must be 'l1', 'mse', or 'huber', got {loss_type!r}")


# =============================================================================
# LEVEL 3 — 2DGS NORMAL CONSISTENCY ADAPTED FOR FLOOR
# =============================================================================

def floor_normal_consistency_loss(
    render_normals:    Tensor,   # [C, H, W, 3]  from rasterization_2dgs
    normals_from_depth: Tensor,  # [C, H, W, 3]  depth-gradient normals
    ground_mask:       Tensor,   # [C, H, W]  bool
    world_up:          Tensor,   # [3]  world-space up vector (usually [0,0,1])
    alpha_map:         Optional[Tensor] = None,  # [C, H, W, 1] accumulated alpha
    world_up_weight:   float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Two-component normal regularisation for floor Gaussians.

    Component A — Standard 2DGS normal consistency (eq. 14 of the paper):
      Minimise 1 - nᵢᵀ·N at every ground pixel, where nᵢ is the splat
      normal and N is the depth-gradient surface normal.  On a flat floor
      both should point straight up.

    Component B — World-up alignment at ground pixels:
      Additionally penalise floor splat normals that deviate from the
      world up vector.  This is a stronger constraint than Component A
      alone because depth-gradient normals can be noisy at flat surfaces
      (small depth differences → large relative noise in the gradient).

    Together they force floor disks to lie flat in the Zw=0 plane:
      - Component A: disk normal ≈ depth gradient normal
      - Component B: disk normal ≈ world up vector
      Both constraints converge to: disk normal = [0, 0, 1] (up)
      which means the disk itself lies in the horizontal plane.

    WHY THIS PREVENTS GREEN GAUSSIANS IN FRONT OF PLAYERS:
      A 2D Gaussian disk with normal [0,0,1] is horizontal.  When the
      rasterizer evaluates it for a player pixel, the ray-disk intersection
      gives a large (u,v) offset from the disk centre, making the Gaussian
      value exp(-(u²+v²)/2) very small.  The disk effectively has zero
      weight at player pixels.  It cannot explain a green color there,
      so the optimizer must assign that green to an actual floor Gaussian,
      not a floating interloper at player depth.

    Args:
        render_normals     : [C, H, W, 3]  rendered splat normals (camera space)
        normals_from_depth : [C, H, W, 3]  depth-gradient normals (camera space)
        ground_mask        : [C, H, W]  True = ground pixel
        world_up           : [3]  world-space up, transform to camera space inside
        alpha_map          : [C, H, W, 1]  optional blending weight map
        world_up_weight    : scale for Component B

    Returns:
        (loss_nc, loss_up) — normal consistency loss and world-up loss.
        Add both to total loss.
    """
    if not ground_mask.any():
        zero = torch.tensor(0.0, device=render_normals.device, requires_grad=True)
        return zero, zero

    # Mask to ground pixels
    mask = ground_mask                         # [C, H, W]
    n_render = render_normals[mask]            # [M, 3]
    n_depth  = normals_from_depth[mask]        # [M, 3]

    # Optional per-pixel alpha weighting
    if alpha_map is not None:
        w = alpha_map[..., 0][mask]            # [M]
        w = w.detach()                         # don't differentiate through alpha
    else:
        w = torch.ones(n_render.shape[0], device=n_render.device)

    # Component A — 2DGS normal consistency (paper eq. 14)
    # 1 - dot(n_render, n_depth) in [0, 2]; 0 = perfectly aligned
    dot_a  = (n_render * n_depth).sum(-1).clamp(-1, 1)   # [M]
    loss_nc = (w * (1.0 - dot_a)).mean()

    # Component B — World-up alignment
    # Transform world_up to camera space (no translation needed for normals)
    # world_up is [3]; we need it in the same space as render_normals
    # render_normals from gsplat are already in world space after the transform
    # in rasterization_2dgs:
    #   render_normals = einsum("...ij,...hwj->...hwi", inv(viewmats)[:,:3,:3], normals_raw)
    # so we compare directly in world space.
    up = F.normalize(world_up.to(n_render.device), dim=0)  # [3]
    dot_b   = (n_render * up.unsqueeze(0)).sum(-1)          # [M]
    loss_up = (w * (1.0 - dot_b.clamp(-1, 1))).mean() * world_up_weight

    return loss_nc, loss_up


# =============================================================================
# COMBINED LOSS — CONVENIENCE WRAPPER
# =============================================================================

def ground_supervision_loss(
    # Rendered outputs
    depths_image:       Tensor,              # [C, H, W, 1]  RGB+ED last channel
    render_normals:     Optional[Tensor],    # [C, H, W, 3]  only for 2DGS
    normals_from_depth: Optional[Tensor],   # [C, H, W, 3]  only for 2DGS
    alpha_map:          Optional[Tensor],   # [C, H, W, 1]
    # Ground prior inputs
    da3_depth:          Optional[Tensor],   # [C, H, W]  scale-aligned DA3 depth
    D_gt:               Optional[Tensor],   # [C, H, W]  from build_ground_depth_map()
    ground_mask:        Tensor,             # [C, H, W]  bool SAM mask
    # Config
    scene_scale:        float = 1.0,
    lambda_da3:         float = 1.0,        # weight for Level 1
    lambda_ground:      float = 2.3,        # weight for Level 2  (paper value)
    lambda_normal:      float = 0.05,       # weight for Level 3  (2DGS paper value)
    lambda_world_up:    float = 0.05,       # weight for world-up component
    world_up:           Optional[Tensor] = None,   # [3] default (0,0,1)
    use_2dgs:           bool = False,       # True if training with 2DGS
    loss_type:          str  = "l1",
) -> Tuple[Tensor, dict]:
    """
    Combined three-level ground supervision.

    Returns (total_ground_loss, breakdown_dict) where breakdown_dict
    contains individual loss components for logging.

    ★ USAGE IN TRAINING LOOP (replace the ground_loss block):

        D_gt = build_ground_depth_map(
            viewmats   = torch.linalg.inv(camtoworlds)[0],   # remove batch dim
            Ks         = Ks[0],
            masks      = ground_masks,   # [C, H, W]
            width      = width,
            height     = height,
        ).unsqueeze(0)   # add C dim → [1, H, W] for C=1

        # For 3DGS:
        gnd_loss, gnd_info = ground_supervision_loss(
            depths_image   = depths_image,          # [1, H, W, 1]
            render_normals = None,
            normals_from_depth = None,
            alpha_map      = None,
            da3_depth      = data["mini_depth"].to(device).unsqueeze(0),
            D_gt           = D_gt,
            ground_mask    = ground_masks,           # [1, H, W]
            scene_scale    = self.scene_scale,
        )

        # For 2DGS (use renders from rasterization_2dgs):
        gnd_loss, gnd_info = ground_supervision_loss(
            depths_image       = render_colors[..., 3:4],
            render_normals     = render_normals,
            normals_from_depth = normals_from_depth,
            alpha_map          = render_alphas,
            da3_depth          = data["mini_depth"].to(device).unsqueeze(0),
            D_gt               = D_gt,
            ground_mask        = ground_masks,
            scene_scale        = self.scene_scale,
            use_2dgs           = True,
        )

        loss = loss + gnd_loss

        # Logging:
        for k, v in gnd_info.items():
            writer.add_scalar(f"train/{k}", v, step)
    """
    total = torch.tensor(0.0, device=depths_image.device, requires_grad=False)
    info  = {}

    # ── Level 1: DA3 dense depth ─────────────────────────────────────────────
    if da3_depth is not None and lambda_da3 > 0:
        l1 = depth_from_da3_loss(
            depths_image = depths_image,
            da3_depth    = da3_depth,
            ground_mask  = ground_mask,
            scene_scale  = scene_scale,
        )
        total = total + l1 * lambda_da3
        info["loss_da3_depth"] = l1.item()

    # ── Level 2: Geometric ground-plane prior ────────────────────────────────
    if D_gt is not None and lambda_ground > 0:
        l2 = ground_plane_prior_loss(
            depths_image = depths_image,
            D_gt         = D_gt,
            loss_type    = loss_type,
        )
        total = total + l2 * lambda_ground
        info["loss_ground_prior"] = l2.item()

    # ── Level 3: Normal consistency (2DGS only) ──────────────────────────────
    if use_2dgs and render_normals is not None and normals_from_depth is not None:
        if world_up is None:
            world_up = torch.tensor([0.0, 0.0, 1.0], device=depths_image.device)

        l3_nc, l3_up = floor_normal_consistency_loss(
            render_normals     = render_normals,
            normals_from_depth = normals_from_depth,
            ground_mask        = ground_mask,
            world_up           = world_up,
            alpha_map          = alpha_map,
            world_up_weight    = lambda_world_up / lambda_normal if lambda_normal > 0 else 1.0,
        )
        total = total + l3_nc * lambda_normal + l3_up * lambda_world_up
        info["loss_normal_consistency"] = l3_nc.item()
        info["loss_world_up"]           = l3_up.item()

    info["loss_ground_total"] = total.item()
    return total, info


# =============================================================================
# SCALE ALIGNMENT — connect DA3 to COLMAP metric
# =============================================================================

def align_da3_to_colmap(
    da3_depth:    Tensor,    # [H, W]  raw DA3 output
    colmap_points: Tensor,   # [M, 2]  pixel coords (u, v)
    colmap_depths: Tensor,   # [M]     sparse COLMAP depths in metres
) -> Tuple[float, float, Tensor]:
    """
    Find scale s and shift t so that  s · da3_depth + t ≈ colmap_depths.

    Uses weighted least-squares (paper eqs. 15-16) on the sparse COLMAP
    depth points projected into the image.

    Returns:
        (s, t, aligned_depth)  where aligned_depth has the same shape as da3_depth.

    USAGE:
        # In the dataloader or pre-processing step (not per iteration):
        s, t, da3_aligned = align_da3_to_colmap(
            da3_depth     = prediction.depth[image_idx],   # [H, W]
            colmap_points = data["points"][0],              # [M, 2]
            colmap_depths = data["depths"][0],              # [M]
        )
        # Store da3_aligned in the dataset so it's loaded as data["mini_depth"]

    WHY NOT PER-ITERATION:
        Scale alignment is a least-squares solve over sparse points.  It
        is cheap (<<1ms) but the result changes every time points change.
        Doing it once per image in a preprocessing pass is cleaner.
        If you want online alignment (adapts as camera poses refine during
        pose_opt), call this each iteration — it is differentiable-free.
    """
    H, W = da3_depth.shape
    # Sample DA3 at COLMAP point locations
    u = colmap_points[:, 0].long().clamp(0, W - 1)
    v = colmap_points[:, 1].long().clamp(0, H - 1)

    x = da3_depth[v, u].float()             # [M]
    y = colmap_depths.float()               # [M]
    w = torch.ones_like(x)

    # Weighted least squares: s, t minimise Σ w(y - sx - t)²
    sw   = w.sum()
    swx  = (w * x).sum()
    swy  = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()

    denom = sw * swxx - swx * swx
    if denom.abs() < 1e-9:
        s, t = 1.0, 0.0
    else:
        s = float(((sw * swxy - swx * swy) / denom).item())
        t = float(((swy - s * swx) / sw).item())

    aligned = (s * da3_depth + t).clamp(min=0.0)
    return s, t, aligned


def get_depth_lambda_schedule(
    step: int,
    max_steps: int,
    strategy: str = "progressive",
    lambda_base: float = 1e-3,
) -> float:
    """
    Progressive scaling of depth loss weight through training phases.
    
    Prevents early overfitting to potentially noisy DA3 depth predictions
    while enforcing geometric constraints later.
    """
    if strategy == "progressive":
        # Phase 1 (0-25%): Minimal depth supervision, focus on color
        if step < max_steps * 0.25:
            return lambda_base * 0.1
        # Phase 2 (25-70%): Rap up depth as reconstruction stabilizes
        elif step < max_steps * 0.70:
            progress = (step - max_steps * 0.25) / (max_steps * 0.45)
            return lambda_base * (0.1 + 0.9 * progress)
        # Phase 3 (70-100%): Full depth weight + ground plane prior
        else:
            return lambda_base
    
    elif strategy == "cosine_warmup":
        # Smooth cosine annealing
        warmup_steps = max_steps * 0.1
        if step < warmup_steps:
            return lambda_base * (1 - torch.cos(torch.tensor(np.pi * step / warmup_steps))) / 2
        return lambda_base
    
    elif strategy == "exponential":
        # Exponential growth toward target
        return lambda_base * (1 - np.exp(-10 * step / max_steps))
    
    return lambda_base

def confidence_weighted_depth_loss(
    depths_image: Tensor,          # [C, H, W, 1] rendered
    da3_depth: Tensor,             # [C, H, W] DA3 prediction
    da3_confidence: Tensor,        # [C, H, W] DA3 confidence (optional)
    colmap_depths: Tensor,         # [C, H, W] sparse COLMAP (optional)
    colmap_valid: Tensor,          # [C, H, W] COLMAP validity mask
    scene_scale: float = 1.0,
    weight_da3: float = 1.0,
    weight_colmap: float = 5.0,    # Trust COLMAP more (sparse but accurate)
) -> Tuple[Tensor, dict]:
    """
    Two-branch weighting: high confidence DA3 pixels + sparse but accurate COLMAP.
    
    COLMAP sparse points should have 5x higher weight because they're metric-accurate.
    DA3 provides dense supervision but with inherent ambiguities.
    """
    device = depths_image.device
    d_pred = depths_image[..., 0]  # [C, H, W]
    
    # Branch 1: Dense DA3 (lower confidence, needs weighting)
    valid_da3 = da3_depth > 0.0
    if da3_confidence is not None:
        # Down-weight low-confidence predictions
        weights_da3 = da3_confidence.clamp(0.0, 1.0)
    else:
        weights_da3 = torch.ones_like(da3_depth)
    
    if valid_da3.any():
        diff_da3 = (d_pred[valid_da3] - da3_depth[valid_da3]).abs()
        loss_da3 = (diff_da3 * weights_da3[valid_da3]).mean()
    else:
        loss_da3 = torch.tensor(0.0, device=device)
    
    # Branch 2: Sparse COLMAP (high confidence, full weight)
    valid_colmap = colmap_valid & (colmap_depths > 0.0)
    if valid_colmap.any():
        diff_colmap = (d_pred[valid_colmap] - colmap_depths[valid_colmap]).abs()
        loss_colmap = diff_colmap.mean()
    else:
        loss_colmap = torch.tensor(0.0, device=device)
    
    total_loss = (weight_da3 * loss_da3 + weight_colmap * loss_colmap) / (weight_da3 + weight_colmap)
    
    info = {
        "loss_da3": loss_da3.item(),
        "loss_colmap": loss_colmap.item(),
        "n_da3_valid": valid_da3.sum().item(),
        "n_colmap_valid": valid_colmap.sum().item(),
    }
    return total_loss * scene_scale, info

def depth_smoothness_loss(
    depths_image: Tensor,      # [C, H, W, 1]
    alphas: Tensor,            # [C, H, W, 1] opacity map
    ground_mask: Tensor,       # [C, H, W] bool
    lambda_smooth: float = 0.01,
) -> Tensor:
    """
    Enforce smooth depth transitions except at object boundaries.
    
    On ground plane: depth should be nearly constant.
    At boundaries: allow sharp depth changes.
    """
    d = depths_image[..., 0]  # [C, H, W]
    
    # Compute depth gradients
    grad_x = torch.abs(d[:, :, :-1] - d[:, :, 1:])
    grad_y = torch.abs(d[:, :-1, :] - d[1:, :, :])
    
    # Down-weight gradients at low-opacity regions (transparent)
    alpha_x = (alphas[:, :, :-1, 0] * alphas[:, :, 1:, 0]).sqrt()
    alpha_y = (alphas[:, :-1, :, 0] * alphas[1:, :, :, 0]).sqrt()
    
    # Mask for ground: smooth depth required
    ground_x = (ground_mask[:, :, :-1] & ground_mask[:, :, 1:])
    ground_y = (ground_mask[:, :-1, :] & ground_mask[1:, :, :])
    
    # Ground pixels should have near-zero depth gradients
    loss_x = (grad_x[ground_x] * alpha_x[ground_x]).mean() if ground_x.any() else 0.0
    loss_y = (grad_y[ground_y] * alpha_y[ground_y]).mean() if ground_y.any() else 0.0
    
    return lambda_smooth * (loss_x + loss_y)

def disparity_depth_loss(
    depths_image: Tensor,      # [C, H, W, 1]
    depths_gt: Tensor,         # [C, H, W]
    valid_mask: Tensor,        # [C, H, W]
    eps: float = 1e-8,
    loss_type: str = "l1",     # "l1" | "berHu" (robust) | "log_diff"
) -> Tensor:
    """
    Depth loss in disparity space: better for large depth ranges.
    
    PlanarGS uses disparity (1/depth) which is more invariant to scale
    and handles both near and far objects better than linear depth.
    """
    d_pred = depths_image[..., 0]
    d_gt = depths_gt
    
    # Convert to disparity (inverse depth)
    disp_pred = torch.where(d_pred > eps, 1.0 / d_pred, torch.zeros_like(d_pred))
    disp_gt = torch.where(d_gt > eps, 1.0 / d_gt, torch.zeros_like(d_gt))
    
    valid = valid_mask & (d_pred > eps) & (d_gt > eps)
    
    if not valid.any():
        return torch.tensor(0.0, device=d_pred.device)
    
    if loss_type == "l1":
        return F.l1_loss(disp_pred[valid], disp_gt[valid])
    
    elif loss_type == "berHu":
        # Huber loss in disparity space (robust to outliers)
        diff = (disp_pred[valid] - disp_gt[valid]).abs()
        threshold = 0.2 * diff.max()
        mask_small = diff < threshold
        loss_small = diff[mask_small].mean() if mask_small.any() else 0.0
        loss_large = (diff[~mask_small]**2 / (2 * threshold)).mean() if (~mask_small).any() else 0.0
        return loss_small + loss_large
    
    elif loss_type == "log_diff":
        # Log difference (handles scale ambiguity)
        return F.l1_loss(torch.log(disp_pred[valid] + eps), 
                        torch.log(disp_gt[valid] + eps))
    
    return F.l1_loss(disp_pred[valid], disp_gt[valid])


def adaptive_depth_lambda(
    losses_history: List[float],
    step: int,
    initial_lambda: float = 1e-3,
    target_loss_ratio: float = 0.05,  # depth_loss / rgb_loss = 5%
) -> float:
    """
    Dynamically adjust depth weight based on running loss statistics.
    
    If depth loss is too dominant → reduce weight.
    If depth loss is negligible → increase weight.
    """
    if len(losses_history) < 100:
        return initial_lambda
    
    recent_depth_losses = losses_history[-100:]
    mean_depth_loss = np.mean(recent_depth_losses)
    
    # Target ratio: depth loss should be ~5% of RGB loss
    # Adjust lambda to maintain this balance
    if mean_depth_loss > target_loss_ratio * 0.5:  # Too high
        return initial_lambda * 0.9
    elif mean_depth_loss < target_loss_ratio * 0.1:  # Too low
        return initial_lambda * 1.1
    
    return initial_lambda

def multiscale_depth_loss(
    renders: Dict[int, Tensor],       # {scale: depth_map}
    depths_gt_dict: Dict[int, Tensor],# {scale: ground_truth}
    valid_masks: Dict[int, Tensor],   # {scale: validity_mask}
    scales_weights: Dict[int, float] = None,  # {scale: weight}
) -> Tuple[Tensor, dict]:
    """
    Supervise depth at multiple scales for coarse-to-fine alignment.
    
    Coarse scales help with global structure, fine scales with detail.
    """
    if scales_weights is None:
        scales_weights = {1: 1.0, 2: 0.5, 4: 0.25}  # Coarse→Fine
    
    total_loss = 0.0
    info = {}
    
    for scale in renders.keys():
        if scale not in depths_gt_dict:
            continue
        
        d_rendered = renders[scale]
        d_gt = depths_gt_dict[scale]
        valid = valid_masks[scale]
        weight = scales_weights.get(scale, 1.0)
        
        if valid.any():
            diff = (d_rendered[valid] - d_gt[valid]).abs()
            loss_scale = diff.mean()
            total_loss = total_loss + weight * loss_scale
            info[f"depth_loss_scale{scale}"] = loss_scale.item()
    
    return total_loss, info

# =============================================================================
# DATASET HELPER — how to load DA3 depth as "mini_depth" in colmap.py Dataset
# =============================================================================
"""
HOW TO ADD DA3 DEPTH TO THE DATASET
─────────────────────────────────────
In datasets/colmap.py, inside Dataset.__getitem__, after loading the image:

    # 1. Load pre-computed aligned DA3 depth (saved as .npy, one per image)
    if self.load_da3_depth:
        image_name = self.parser.image_names[index]
        stem = os.path.splitext(image_name)[0]
        da3_path = os.path.join(self.da3_depth_dir, stem + ".npy")
        if os.path.exists(da3_path):
            da3 = np.load(da3_path).astype(np.float32)   # [H_full, W_full]
            # Resize to match the downsampled image
            h, w = image.shape[:2]
            da3 = cv2.resize(da3, (w, h), interpolation=cv2.INTER_LINEAR)
            data["mini_depth"] = torch.from_numpy(da3)   # [H, W]
        else:
            # No DA3 for this image — return zeros (loss will skip these)
            data["mini_depth"] = torch.zeros(image.shape[:2], dtype=torch.float32)

HOW TO PRE-COMPUTE THE .npy FILES (run once before training):

    from depth_anything_3.api import DepthAnything3
    from ground_plane_losses import align_da3_to_colmap

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to("cuda")

    prediction = model.inference(
        images,
        extrinsics=extrinsics,      # from COLMAP
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
    )

    for i, (depth, name) in enumerate(zip(prediction.depth, image_names)):
        # depth is [H, W] float32, already metric if align_to_input_ext_scale=True
        # But double-check with COLMAP sparse points for extra robustness:
        s, t, aligned = align_da3_to_colmap(
            da3_depth     = torch.from_numpy(depth),
            colmap_points = colmap_point_pixels[i],   # [M, 2]
            colmap_depths = colmap_depth_values[i],   # [M]
        )
        stem = os.path.splitext(name)[0]
        np.save(f"/data/da3_aligned/{stem}.npy", aligned.numpy())
"""

"""
ground_plane_guided.py
=======================
Ground-plane depth prior module.

Implements Algorithm 1 from:
  "Scene-Constrained Neural Radiance Fields for High-Quality Sports Scene
   Rendering Based on Visual Sensor Network" (Dai et al. 2024)

and the scale-alignment utility from eqs. (15–16) used with DA3 depth maps.

Public API
----------
  quat_to_rotmat            — quaternion → 3×3 rotation
  quat_scale_to_covar       — (q, s) → 3D covariance Σ
  world_to_cam              — world-space Gaussians → camera space
  persp_proj                — EWA perspective projection → 2D
  isect_tiles               — Gaussian–tile intersection list
  isect_offset_encode       — per-tile start-index lookup table
  cam2world_ground          — camera rays → Zw=0 world points
  calculate_depth           — Euclidean camera-to-ground-point distance
  ground_plane_depth_guided — full Algorithm 1 pipeline
  depth_supervision_loss    — pixel-wise Ldist (paper eq. 13–14)
  align_depth_scale         — weighted least-squares scale/shift for DA3
"""

import math
import struct
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# SECTION 1 — GEOMETRY HELPERS
# =============================================================================

def quat_to_rotmat(quats: Tensor) -> Tensor:
    """
    Quaternion (w, x, y, z) → 3×3 rotation matrix.
    Input  : [..., 4]
    Output : [..., 3, 3]
    """
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack([
        1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),          1 - 2*(x**2 + z**2),  2*(y*z - w*x),
        2*(x*z - w*y),          2*(y*z + w*x),         1 - 2*(x**2 + y**2),
    ], dim=-1)
    return R.reshape(quats.shape[:-1] + (3, 3))


def quat_scale_to_covar(
    quats:  Tensor,           # [..., 4]
    scales: Tensor,           # [..., 3]
    triu:   bool = False,
) -> Tensor:
    """
    Build 3D covariance  Σ = R · diag(s)² · Rᵀ  from quaternion + scale.

    triu=True  → [..., 6]   upper-triangle elements
    triu=False → [..., 3, 3] full symmetric matrix
    """
    R = quat_to_rotmat(quats)                         # [..., 3, 3]
    M = R * scales[..., None, :]                      # R · diag(s)
    covars = torch.einsum("...ij,...kj->...ik", M, M) # [..., 3, 3]
    if triu:
        c = covars.reshape(quats.shape[:-1] + (9,))
        covars = (
            c[..., [0, 1, 2, 4, 5, 8]]
            + c[..., [0, 3, 6, 4, 7, 8]]
        ) / 2.0
    return covars


# =============================================================================
# SECTION 2 — WORLD → CAMERA
# =============================================================================

def world_to_cam(
    means:    Tensor,   # [..., N, 3]
    covars:   Tensor,   # [..., N, 3, 3]
    viewmats: Tensor,   # [..., C, 4, 4]  world-to-camera
) -> Tuple[Tensor, Tensor]:
    """
    Apply extrinsic [R|t] to Gaussian means and covariances.

    Returns
    -------
    means_c  : [..., C, N, 3]
    covars_c : [..., C, N, 3, 3]
    """
    R = viewmats[..., :3, :3]   # [..., C, 3, 3]
    t = viewmats[..., :3,  3]   # [..., C, 3]

    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means)
        + t[..., None, :]
    )
    covars_c = torch.einsum("...cij,...njk,...clk->...cnil", R, covars, R)
    return means_c, covars_c


# =============================================================================
# SECTION 3 — EWA PERSPECTIVE PROJECTION
# =============================================================================

def persp_proj(
    means:   Tensor,    # [..., C, N, 3]
    covars:  Tensor,    # [..., C, N, 3, 3]
    Ks:      Tensor,    # [..., C, 3, 3]
    width:   int,
    height:  int,
    eps2d:   float = 0.3,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Project 3D Gaussians to 2D image plane using EWA splatting.

    Returns
    -------
    means2d : [..., C, N, 2]
    cov2d   : [..., C, N, 2, 2]
    conics  : [..., C, N, 3]    — (a, b, c) of inverse 2D cov
    depths  : [..., C, N]       — Zc (optical-axis depth)
    radii   : [..., C, N, 2]    — pixel radii for tile intersection
    """
    tx, ty, tz = torch.unbind(means, dim=-1)
    tz2 = tz ** 2

    fx  = Ks[..., 0, 0, None];  fy  = Ks[..., 1, 1, None]
    cx  = Ks[..., 0, 2, None];  cy  = Ks[..., 1, 2, None]

    tan_fovx = 0.5 * width  / fx
    tan_fovy = 0.5 * height / fy
    lim_xp = (width  - cx) / fx + 0.3 * tan_fovx
    lim_xn =           cx  / fx + 0.3 * tan_fovx
    lim_yp = (height - cy) / fy + 0.3 * tan_fovy
    lim_yn =           cy  / fy + 0.3 * tan_fovy

    tx = tz * torch.clamp(tx / tz, -lim_xn, lim_xp)
    ty = tz * torch.clamp(ty / tz, -lim_yn, lim_yp)

    O = torch.zeros_like(tz)
    J = torch.stack([
        fx / tz, O, -fx * tx / tz2,
        O, fy / tz, -fy * ty / tz2,
    ], dim=-1).reshape(means.shape[:-1] + (2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    cov2d = cov2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d
    det   = (cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2).clamp(1e-10)

    conics = torch.stack([
        cov2d[..., 1, 1] / det,
        -(cov2d[..., 0, 1] + cov2d[..., 1, 0]) / 2.0 / det,
        cov2d[..., 0, 0] / det,
    ], dim=-1)

    means2d = torch.einsum("...ij,...nj->...ni", Ks[..., :2, :3], means)
    means2d = means2d / tz[..., None]

    radii = torch.stack([
        torch.ceil(3.33 * torch.sqrt(cov2d[..., 0, 0])),
        torch.ceil(3.33 * torch.sqrt(cov2d[..., 1, 1])),
    ], dim=-1)

    return means2d, cov2d, conics, tz, radii


# =============================================================================
# SECTION 4 — TILE INTERSECTION + SORTING
# =============================================================================

@torch.no_grad()
def isect_tiles(
    means2d:     Tensor,   # [..., N, 2]
    radii:       Tensor,   # [..., N, 2]
    depths:      Tensor,   # [..., N]
    tile_size:   int,
    tile_width:  int,
    tile_height: int,
    sort:        bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Emit one (tile, depth) record per (Gaussian, tile) intersection.
    Records are sorted by a packed int64 key so depth order is front-to-back
    within each tile.

    Returns
    -------
    tiles_per_gauss : [..., N]      tiles touched by each Gaussian
    isect_ids       : [n_isects]    packed 64-bit sort key
    flatten_ids     : [n_isects]    (image, gauss) flat index per record
    """
    image_dims = means2d.shape[:-2]
    N  = means2d.shape[-2]
    I  = math.prod(image_dims)
    dev = means2d.device

    means2d = means2d.reshape(I, N, 2)
    radii   = radii.reshape(I, N, 2)
    depths  = depths.reshape(I, N)

    tile_m = means2d / tile_size
    tile_r = radii   / tile_size
    tile_mins = torch.floor(tile_m - tile_r).int().clamp(
        torch.tensor([0, 0], device=dev),
        torch.tensor([tile_width, tile_height], device=dev),
    )
    tile_maxs = torch.ceil(tile_m + tile_r).int().clamp(
        torch.tensor([0, 0], device=dev),
        torch.tensor([tile_width, tile_height], device=dev),
    )

    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)
    tiles_per_gauss *= (radii > 0.0).all(dim=-1)

    n_isects     = int(tiles_per_gauss.sum().item())
    isect_ids_lo = torch.empty(n_isects, dtype=torch.int32, device=dev)
    isect_ids_hi = torch.empty(n_isects, dtype=torch.int32, device=dev)
    flatten_ids  = torch.empty(n_isects, dtype=torch.int32, device=dev)

    cum_tiles    = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    image_n_bits = I.bit_length()
    tile_n_bits  = (tile_width * tile_height).bit_length()
    assert image_n_bits + tile_n_bits + 32 <= 64, \
        "Too many images/tiles to fit in 64-bit key"

    def _kernel(img: int, g: int):
        if radii[img, g, 0] <= 0.0:
            return
        idx      = img * N + g
        cur      = int(cum_tiles[idx - 1].item()) if idx > 0 else 0
        depth_id = int(struct.unpack("i", struct.pack("f", depths[img, g].item()))[0]) & 0xFFFFFFFF
        t_min    = tile_mins[img, g]
        t_max    = tile_maxs[img, g]
        for y in range(int(t_min[1].item()), int(t_max[1].item())):
            for x in range(int(t_min[0].item()), int(t_max[0].item())):
                tile_id         = y * tile_width + x
                isect_ids_lo[cur] = depth_id
                isect_ids_hi[cur] = (img << tile_n_bits) | tile_id
                flatten_ids[cur]  = idx
                cur += 1

    for img in range(I):
        for g in range(N):
            _kernel(img, g)

    isect_ids = (isect_ids_hi.to(torch.int64) << 32) | (
        isect_ids_lo.to(torch.int64) & 0xFFFFFFFF
    )
    if sort:
        isect_ids, sort_idx = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_idx]

    tiles_per_gauss = tiles_per_gauss.reshape(image_dims + (N,)).int()
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids:   Tensor,
    I:           int,
    tile_width:  int,
    tile_height: int,
) -> Tensor:
    """
    Convert sorted intersection list → per-tile start-index lookup table.
    offsets[img, y, x] = start index in isect_ids for tile (img, x, y).
    """
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros(
        (I, tile_height, tile_width), dtype=torch.int64, device=isect_ids.device
    )
    ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)
    img_ids  = ids_uq >> tile_n_bits
    tile_ids = ids_uq & ((1 << tile_n_bits) - 1)
    tx = tile_ids % tile_width
    ty = tile_ids // tile_width
    tile_counts[img_ids, ty, tx] = counts
    cum = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    return (cum - tile_counts).int()


# =============================================================================
# SECTION 5 — GROUND-PLANE RAY INTERSECTION  (paper eq. 7–8 / Algorithm 1)
# =============================================================================

def cam2world_ground(
    means_2d: Tensor,   # [..., C, N, 2]  normalised camera rays (Xc/Zc, Yc/Zc)
    viewmats: Tensor,   # [..., C, 4, 4]  world-to-camera
) -> Tensor:
    """
    Back-project camera-space rays onto the Zw = 0 ground plane (eq. 7).

    Solves a 2×2 linear system per ray:
        A · [Xw, Yw]ᵀ = b
    where A and b depend on the rotation R and translation t of the viewmat.

    Returns [..., C, N, 2] — (Xw, Yw) on the ground plane.
    """
    x_c, y_c = torch.unbind(means_2d, dim=-1)   # [..., C, N]

    R  = viewmats[..., :3, :3]
    t  = viewmats[..., :3,  3]

    r11 = R[..., 0, 0]; r12 = R[..., 0, 1]
    r21 = R[..., 1, 0]; r22 = R[..., 1, 1]
    r31 = R[..., 2, 0]; r32 = R[..., 2, 1]
    t1  = t[..., 0];    t2  = t[..., 1];  t3 = t[..., 2]

    # Broadcast rotation scalars over N rays
    A11 = x_c * r31[..., None] - r11[..., None]
    A12 = x_c * r32[..., None] - r12[..., None]
    A21 = y_c * r31[..., None] - r21[..., None]
    A22 = y_c * r32[..., None] - r22[..., None]

    A = torch.stack([
        torch.stack([A11, A12], dim=-1),
        torch.stack([A21, A22], dim=-1),
    ], dim=-2)   # [..., C, N, 2, 2]

    b = torch.stack([
        t1[..., None] - x_c * t3[..., None],
        t2[..., None] - y_c * t3[..., None],
    ], dim=-1).unsqueeze(-1)   # [..., C, N, 2, 1]

    Xw = torch.linalg.solve(A, b).squeeze(-1)   # [..., C, N, 2]
    return Xw


def calculate_depth(
    Xw:       Tensor,   # [..., C, N, 2]  (Xw, Yw) on Zw=0 plane
    viewmats: Tensor,   # [..., C, 4, 4]
) -> Tensor:
    """
    Euclidean distance from camera centre to each ground-plane point.
    (Algorithm 1, line 11:  depth = ‖pw − t_cam‖₂)

    Returns [..., C, N].
    """
    R       = viewmats[..., :3, :3]
    t_cam   = viewmats[..., :3,  3]

    # Camera centre in world space:  c = −Rᵀ t_cam
    cam_c   = -torch.einsum("...ji,...j->...i", R, t_cam)   # [..., C, 3]
    cam_c   = cam_c[..., None, :]                            # [..., C, 1, 3]

    zeros   = torch.zeros_like(Xw[..., :1])
    Xw_3d   = torch.cat([Xw, zeros], dim=-1)                 # [..., C, N, 3]

    return torch.linalg.norm(cam_c - Xw_3d, dim=-1)          # [..., C, N]


# =============================================================================
# SECTION 6 — FULL GROUND-PLANE PIPELINE  (Algorithm 1)
# =============================================================================

def ground_plane_depth_guided(
    means:      Tensor,    # [..., N, 3]    Gaussian world-space centres
    covars:     Tensor,    # [..., N, 3, 3] Gaussian world-space covariances
    viewmats:   Tensor,    # [..., C, 4, 4] world-to-camera extrinsics
    Ks:         Tensor,    # [..., C, 3, 3] camera intrinsics
    masks:      Tensor,    # [..., C, H, W] bool — True = ground pixel
    width:      int,
    height:     int,
    near_plane: float = 0.01,
    far_plane:  float = 1e10,
    eps2d:      float = 0.3,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Full ground-plane depth prior pipeline (Algorithm 1, Dai et al. 2024).

    Steps
    -----
    1. World → camera                  (world_to_cam)
    2. EWA projection → conics, radii  (persp_proj)
    3. Near/far + frustum validity filter
    4. For each masked (ground) pixel:
         a. Unproject to normalised camera ray  (eq. 8)
         b. Intersect with Zw = 0              (cam2world_ground)
         c. Euclidean depth                    (calculate_depth)
    5. Return depth map D_gt

    Returns
    -------
    means2d   : [..., C, N, 2]
    conics    : [..., C, N, 3]
    depths_zc : [..., C, N]       Zc depths (used for tile sorting)
    radii     : [..., C, N, 2]   (int, 0 for invalid Gaussians)
    D_gt      : [..., C, H, W]   ground-plane depth (0 = no ground)
    valid     : [..., C, N]      bool — Gaussians that passed near/far
    """
    # Step 1–2
    means_c, covars_c = world_to_cam(means, covars, viewmats)
    means2d, cov2d, conics, depths_zc, radii = persp_proj(
        means_c, covars_c, Ks, width, height, eps2d
    )

    # Step 3: validity filter
    valid      = (depths_zc > near_plane) & (depths_zc < far_plane)
    radii_out  = radii.clone()
    radii_out[~valid] = 0
    inside = (
        (means2d[..., 0] + radii_out[..., 0] > 0)
        & (means2d[..., 0] - radii_out[..., 0] < width)
        & (means2d[..., 1] + radii_out[..., 1] > 0)
        & (means2d[..., 1] - radii_out[..., 1] < height)
    )
    radii_out[~inside] = 0

    # Step 4: per-camera ground-plane depth map
    fx = Ks[..., 0, 0];  fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2];  cy = Ks[..., 1, 2]

    batch_dims = means.shape[:-2]
    C = viewmats.shape[-3]

    D_gt = torch.zeros(
        batch_dims + (C, height, width),
        device=means.device, dtype=means.dtype,
    )

    for c in range(C):
        mask_c = masks[..., c, :, :]           # [..., H, W]
        fx_c   = fx[..., c].squeeze()
        fy_c   = fy[..., c].squeeze()
        cx_c   = cx[..., c].squeeze()
        cy_c   = cy[..., c].squeeze()
        vm_c   = viewmats[..., c:c+1, :, :]   # [..., 1, 4, 4]

        # Ground pixel coordinates
        mask_2d = mask_c[0] if mask_c.dim() > 2 else mask_c
        ys, xs  = torch.where(mask_2d)
        if xs.numel() == 0:
            continue

        # Eq. (8): pinhole un-projection → normalised camera ray
        Xc = (xs.float() - cx_c) / fx_c    # [M]
        Yc = (ys.float() - cy_c) / fy_c    # [M]

        rays = torch.stack([Xc, Yc], dim=-1)[None, None, :, :]  # [1, 1, M, 2]

        # Eq. (7): ray–ground-plane intersection
        Xw_pts    = cam2world_ground(rays, vm_c)    # [1, 1, M, 2]
        depth_pts = calculate_depth(Xw_pts, vm_c)  # [1, 1, M]
        depth_vals = depth_pts.squeeze([0, 1])      # [M]

        valid_depth  = (depth_vals > near_plane) & (depth_vals < far_plane)
        valid_bounds = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        valid_pts    = valid_depth & valid_bounds

        if valid_pts.any():
            D_gt[..., c, ys[valid_pts], xs[valid_pts]] = depth_vals[valid_pts]

    return means2d, conics, depths_zc, radii_out.int(), D_gt, valid


# =============================================================================
# SECTION 7 — LOSSES
# =============================================================================

def depth_supervision_loss(
    D_pred:  Tensor,   # [..., C, N]      rendered Gaussian depths
    D_gt:    Tensor,   # [..., C, H, W]   ground-plane depth map
    means2d: Tensor,   # [..., C, N, 2]   projected Gaussian centres
    width:   int,
    height:  int,
) -> Tensor:
    """
    Partial depth supervision — paper eq. (13–14):

        Ldist(r) = MSE(D̂(r), D(r))   if D(r) > 0
                   0                   otherwise

    Samples D_gt at each Gaussian's projected pixel centre, then computes
    MSE only where D_gt > 0 (i.e. the pixel is labelled ground).
    """
    norm_x = (means2d[..., 0] / (width  - 1)) * 2 - 1
    norm_y = (means2d[..., 1] / (height - 1)) * 2 - 1
    grid   = torch.stack([norm_x, norm_y], dim=-1)    # [..., C, N, 2]

    D_gt_4d   = D_gt.unsqueeze(-3)
    orig_shape = grid.shape[:-1]
    B  = math.prod(orig_shape[:-1])
    N  = orig_shape[-1]

    D_sampled = F.grid_sample(
        D_gt_4d.reshape(B, 1, height, width),
        grid.reshape(B, 1, N, 2),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).reshape(orig_shape)

    valid = D_sampled > 0.0
    if not valid.any():
        return torch.tensor(0.0, device=D_pred.device, requires_grad=True)
    return F.mse_loss(D_pred[valid], D_sampled[valid])


def color_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """L2 colour reconstruction loss — eq. (11)."""
    return F.mse_loss(rendered, target)


def total_loss(
    rendered:     Tensor,
    target:       Tensor,
    D_pred:       Tensor,
    D_gt:         Tensor,
    means2d:      Tensor,
    width:        int,
    height:       int,
    lambda_dist:  float = 2.3,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Full training objective  L = Lcolor + λ · Ldist  (eq. 14).
    """
    Lcolor = color_loss(rendered, target)
    Ldist  = depth_supervision_loss(D_pred, D_gt, means2d, width, height)
    L      = Lcolor + lambda_dist * Ldist
    return L, Lcolor, Ldist


# =============================================================================
# SECTION 8 — SCALE ALIGNMENT  (eqs. 15–16, DA3 / ZoeDepth)
# =============================================================================

def align_depth_scale(
    D_pre:    Tensor,                      # [...] monocular depth estimate
    D_sparse: Tensor,                      # [...] sparse metric GT (D_gt > 0 pixels)
    weights:  Optional[Tensor] = None,     # [...] confidence weights ∈ [0, 1]
) -> Tuple[float, float, Tensor]:
    """
    Fit scale s and shift t  such that  s·D_pre + t ≈ D_sparse,
    using weighted least squares on pixels where D_sparse > 0.

    Paper eqs. (15-16):
        s = (sw*swxy - swx*swy) / (sw*swxx - swx^2)
        t = (swy - s*swx) / sw

    Returns
    -------
    s     : float
    t     : float
    D_reg : Tensor   s·D_pre + t  (metric-aligned depth map)
    """
    valid = D_sparse > 0
    x = D_pre[valid].float()
    y = D_sparse[valid].float()
    w = weights[valid].float() if weights is not None else torch.ones_like(x)

    sw   = w.sum()
    swx  = (w * x).sum()
    swy  = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()

    denom = sw * swxx - swx * swx
    if denom.abs() < 1e-9 or valid.sum() < 2:
        s, t = 1.0, 0.0
    else:
        s = float(((sw * swxy - swx * swy) / denom).item())
        t = float(((swy - s * swx) / sw).item())

    D_reg = s * D_pre + t
    return s, t, D_reg