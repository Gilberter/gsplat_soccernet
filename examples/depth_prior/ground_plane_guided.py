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