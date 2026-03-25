"""
Scene-Constrained Gaussian Splatting
=====================================
Full PyTorch implementation combining:
  - 3D Gaussian Splatting (Kerbl et al. 2023)  — forward rendering pipeline
  - Scene-Constrained NeRF (Dai et al. 2024)   — ground-plane depth supervision

Pipeline overview
-----------------
1.  Quaternion + scale  →  3D covariance Σ          (_quat_scale_to_covar)
2.  World → camera      →  means_c, covars_c         (_world_to_cam)
3.  Camera → 2D         →  means2d, cov2d, conics    (_persp_proj + conic)
4.  Depth from Zc       →  depths                    (from means_c[...,2])
5.  Tile intersection   →  sorted (tile, depth) list (_isect_tiles)
6.  Offset table        →  per-tile start indices    (_isect_offset_encode)
7.  Ground depth prior  →  D_gt per masked pixel     (ground_plane_depth_guided)
8.  Training loss       →  Lcolor + λ·Ldist          (total_loss)
"""

import math
import struct
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# SECTION 1 — GEOMETRY HELPERS
# =============================================================================

def quat_to_rotmat(quats: Tensor) -> Tensor:
    """
    Quaternion (w, x, y, z) → 3x3 rotation matrix.
    Input: [..., 4]
    Output: [..., 3, 3]
    """
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack([
        1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2),
    ], dim=-1)
    return R.reshape(quats.shape[:-1] + (3, 3))


def quat_scale_to_covar(
    quats: Tensor,   # [..., 4]
    scales: Tensor,  # [..., 3]
    triu: bool = False,
) -> Tensor:
    """
    Build 3D covariance  Σ = R · S · Sᵀ · Rᵀ  from quaternion and scale.

    Paper context: each Gaussian stores (q, s) instead of Σ directly to
    guarantee positive-definiteness during optimisation.

    triu=True returns the 6 upper-triangle elements [..., 6] instead of
    the full [..., 3, 3] matrix.
    """
    R = quat_to_rotmat(quats)          # [..., 3, 3]
    M = R * scales[..., None, :]       # [..., 3, 3]  =  R·diag(s)
    covars = torch.einsum("...ij,...kj->...ik", M, M)  # [..., 3, 3]
    if triu:
        covars = covars.reshape(quats.shape[:-1] + (9,))
        covars = (
            covars[..., [0, 1, 2, 4, 5, 8]]
            + covars[..., [0, 3, 6, 4, 7, 8]]
        ) / 2.0  # [..., 6]
    return covars


# =============================================================================
# SECTION 2 — WORLD → CAMERA
# =============================================================================

def world_to_cam(
    means: Tensor,    # [..., N, 3]
    covars: Tensor,   # [..., N, 3, 3]
    viewmats: Tensor, # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """
    Apply extrinsic transform E = [R | t] to Gaussian means and covariances.

    Paper eq. (5):  Pc = Ti · Pw
    where Ti = Pi · M unifies all cameras into the world frame (Zw=0 ground).

    means_c  = R · means + t       [..., C, N, 3]
    covars_c = R · Σ · Rᵀ          [..., C, N, 3, 3]
    """
    R = viewmats[..., :3, :3]   # [..., C, 3, 3]
    t = viewmats[..., :3, 3]    # [..., C, 3]

    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means)
        + t[..., None, :]
    )  # [..., C, N, 3]

    covars_c = torch.einsum(
        "...cij,...njk,...clk->...cnil", R, covars, R
    )  # [..., C, N, 3, 3]

    return means_c, covars_c


# =============================================================================
# SECTION 3 — CAMERA → 2D PROJECTION  (EWA splatting)
# =============================================================================

def persp_proj(
    means: Tensor,   # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,      # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Project 3D Gaussians to 2D image plane using EWA splatting.

    Returns
    -------
    means2d   : [..., C, N, 2]   pixel-space projected centers
    cov2d     : [..., C, N, 2, 2] projected 2D covariance
    conics    : [..., C, N, 3]   (a, b, c) = upper-tri of (Σ'2D)⁻¹
    depths    : [..., C, N]      Zc — depth along optical axis
    radii     : [..., C, N, 2]   pixel radius for tile intersection
    """
    tx, ty, tz = torch.unbind(means, dim=-1)
    tz2 = tz ** 2

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]

    # Frustum clamping — prevents numerical explosion at image edges
    tan_fovx = 0.5 * width  / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width  - cx) / fx + 0.3 * tan_fovx
    lim_x_neg =           cx  / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg =           cy  / fy + 0.3 * tan_fovy

    tx = tz * torch.clamp(tx / tz, -lim_x_neg, lim_x_pos)
    ty = tz * torch.clamp(ty / tz, -lim_y_neg, lim_y_pos)

    # Jacobian of the perspective map (affine approximation around mean)
    # J = [[fx/tz,  0,      -fx·tx/tz²],
    #      [0,      fy/tz,  -fy·ty/tz²]]
    O = torch.zeros_like(tz)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2,
         O, fy / tz, -fy * ty / tz2],
        dim=-1,
    ).reshape(means.shape[:-1] + (2, 3))  # [..., C, N, 2, 3]

    # 2D covariance:  Σ'2D = J · Σ3D · Jᵀ
    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))

    # Add eps2d to diagonal for numerical stability (anti-aliasing compensation)
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2
    cov2d = cov2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d
    det = (cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2).clamp(1e-10)

    # Conic = (Σ'2D)⁻¹  stored as (a, b, c) for the quadratic form
    # power = -½(a·Δx² + 2b·Δx·Δy + c·Δy²)
    # a = Σ22/det,  b = -Σ12/det,  c = Σ11/det
    conics = torch.stack([
        cov2d[..., 1, 1] / det,
        -(cov2d[..., 0, 1] + cov2d[..., 1, 0]) / 2.0 / det,
        cov2d[..., 0, 0] / det,
    ], dim=-1)  # [..., C, N, 3]

    # 2D pixel-space centers:  u = fx·Xc/Zc + cx,  v = fy·Yc/Zc + cy
    means2d = torch.einsum("...ij,...nj->...ni", Ks[..., :2, :3], means)
    means2d = means2d / tz[..., None]  # [..., C, N, 2]

    depths = tz  # [..., C, N]  — Zc, depth along optical axis

    # Screen-space radius ≈ 3.33σ (covers ~99.9% of Gaussian mass)
    radius_x = torch.ceil(3.33 * torch.sqrt(cov2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(cov2d[..., 1, 1]))
    radii = torch.stack([radius_x, radius_y], dim=-1)  # [..., C, N, 2]

    return means2d, cov2d, conics, depths, radii


# =============================================================================
# SECTION 4 — TILE INTERSECTION + SORTING
# =============================================================================

@torch.no_grad()
def isect_tiles(
    means2d: Tensor,   # [..., N, 2]
    radii: Tensor,     # [..., N, 2]
    depths: Tensor,    # [..., N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    For each (Gaussian, tile) intersection emit one record.
    Records are sorted by (image_id | tile_id | depth) packed into int64.

    IEEE-754 trick: positive float bit-patterns sort identically to
    the float values, so a plain integer sort = front-to-back depth sort.

    Returns
    -------
    tiles_per_gauss : [..., N]       how many tiles each Gaussian touches
    isect_ids       : [n_isects]     packed 64-bit sort key
    flatten_ids     : [n_isects]     which (image, gauss) index each entry is
    """
    image_dims = means2d.shape[:-2]
    N = means2d.shape[-2]
    I = math.prod(image_dims)
    device = means2d.device

    means2d = means2d.reshape(I, N, 2)
    radii   = radii.reshape(I, N, 2)
    depths  = depths.reshape(I, N)

    tile_means2d = means2d / tile_size
    tile_radii   = radii   / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil (tile_means2d + tile_radii).int()
    tile_mins[..., 0] = tile_mins[..., 0].clamp(0, tile_width)
    tile_mins[..., 1] = tile_mins[..., 1].clamp(0, tile_height)
    tile_maxs[..., 0] = tile_maxs[..., 0].clamp(0, tile_width)
    tile_maxs[..., 1] = tile_maxs[..., 1].clamp(0, tile_height)

    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)
    tiles_per_gauss *= (radii > 0.0).all(dim=-1)

    n_isects = int(tiles_per_gauss.sum().item())
    isect_ids_lo = torch.empty(n_isects, dtype=torch.int32, device=device)
    isect_ids_hi = torch.empty(n_isects, dtype=torch.int32, device=device)
    flatten_ids  = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    image_n_bits = I.bit_length()
    tile_n_bits  = (tile_width * tile_height).bit_length()
    assert image_n_bits + tile_n_bits + 32 <= 64, \
        "Too many images or tiles to pack into 64 bits"

    def kernel(image_id: int, gauss_id: int):
        if radii[image_id, gauss_id, 0] <= 0.0:
            return
        index    = image_id * N + gauss_id
        curr_idx = int(cum_tiles[index - 1].item()) if index > 0 else 0

        # Reinterpret float32 depth bits as int32 (preserves sort order)
        depth_f32 = depths[image_id, gauss_id].item()
        depth_id  = struct.unpack("i", struct.pack("f", depth_f32))[0]
        depth_id  = int(depth_id) & 0xFFFFFFFF

        t_min = tile_mins[image_id, gauss_id]
        t_max = tile_maxs[image_id, gauss_id]
        for y in range(int(t_min[1].item()), int(t_max[1].item())):
            for x in range(int(t_min[0].item()), int(t_max[0].item())):
                tile_id = y * tile_width + x
                isect_ids_lo[curr_idx] = depth_id
                isect_ids_hi[curr_idx] = (image_id << tile_n_bits) | tile_id
                flatten_ids [curr_idx] = index
                curr_idx += 1

    for img in range(I):
        for g in range(N):
            kernel(img, g)

    isect_ids = (
        isect_ids_hi.to(torch.int64) << 32
    ) | (isect_ids_lo.to(torch.int64) & 0xFFFFFFFF)

    if sort:
        isect_ids, sort_idx = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_idx]

    tiles_per_gauss = tiles_per_gauss.reshape(image_dims + (N,)).int()
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor,
    I: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """
    Convert sorted intersection list into a per-tile offset lookup table.

    offsets[img, y, x] = index in isect_ids where tile (img, x, y) begins.
    The renderer reads isect_ids[offset : offset + count] for each tile.
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
# SECTION 5 — GROUND-PLANE DEPTH  (Paper Algorithm 1)
# =============================================================================

def cam2world_ground(
    means_2d: Tensor,  # [..., C, N, 2]  normalised camera rays (Xc/Zc, Yc/Zc)
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tensor:
    """
    Back-project camera-space rays onto the Zw = 0 ground plane.

    Paper eq. (7): substitute Zw=0 into the world→camera mapping,
    eliminate the unknown depth d, giving a 2x2 linear system in (Xw, Yw).

    A · [Xw, Yw]ᵀ = b

    where
      A = [[Xc·r31 - r11,  Xc·r32 - r12],
           [Yc·r31 - r21,  Yc·r32 - r22]]
      b = [t1 - Xc·t3,
           t2 - Yc·t3]

    Returns: [..., C, N, 2]  — (Xw, Yw) world coordinates on the ground plane
    """
    x_c, y_c = torch.unbind(means_2d, dim=-1)  # [..., C, N]

    R  = viewmats[..., :3, :3]   # [..., C, 3, 3]
    t  = viewmats[..., :3, 3]    # [..., C, 3]

    r11 = R[..., 0, 0]; r12 = R[..., 0, 1]
    r21 = R[..., 1, 0]; r22 = R[..., 1, 1]
    r31 = R[..., 2, 0]; r32 = R[..., 2, 1]
    t1  = t[..., 0];    t2  = t[..., 1];  t3 = t[..., 2]

    # Broadcast rotation scalars over N Gaussians
    A11 = x_c * r31[..., None] - r11[..., None]
    A12 = x_c * r32[..., None] - r12[..., None]
    A21 = y_c * r31[..., None] - r21[..., None]
    A22 = y_c * r32[..., None] - r22[..., None]

    A = torch.stack([
        torch.stack([A11, A12], dim=-1),
        torch.stack([A21, A22], dim=-1),
    ], dim=-2)  # [..., C, N, 2, 2]

    b1 = t1[..., None] - x_c * t3[..., None]
    b2 = t2[..., None] - y_c * t3[..., None]
    b  = torch.stack([b1, b2], dim=-1)[..., None]  # [..., C, N, 2, 1]

    Xw = torch.linalg.solve(A, b).squeeze(-1)  # [..., C, N, 2]
    return Xw


def calculate_depth(
    Xw: Tensor,        # [..., C, N, 2]  (Xw, Yw) on Zw=0 plane
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tensor:
    """
    Compute Euclidean distance from camera centre to ground-plane point.

    Paper Algorithm 1, line 11:  depth = ‖pw − t‖₂

    Camera centre in world space: c = −Rᵀ · t_cam
    (inverse of the extrinsic: t_cam maps world origin → camera space)

    Returns: [..., C, N]
    """
    R      = viewmats[..., :3, :3]
    t_cam  = viewmats[..., :3, 3]

    # Camera centre: −Rᵀ t  (rotation is orthogonal so Rᵀ = R⁻¹)
    cam_center = -torch.einsum("...ji,...j->...i", R, t_cam)  # [..., C, 3]
    cam_center = cam_center[..., None, :]                      # [..., C, 1, 3]

    # Lift (Xw, Yw) to 3D by appending Zw = 0
    zeros  = torch.zeros_like(Xw[..., :1])
    Xw_3d  = torch.cat([Xw, zeros], dim=-1)                   # [..., C, N, 3]

    depth = torch.linalg.norm(cam_center - Xw_3d, dim=-1)     # [..., C, N]
    return depth


def ground_plane_depth_guided(
    means: Tensor,     # [..., N, 3]    Gaussian world-space centres
    covars: Tensor,    # [..., N, 3, 3] Gaussian world-space covariances
    viewmats: Tensor,  # [..., C, 4, 4] extrinsic matrices (world → camera)
    Ks: Tensor,        # [..., C, 3, 3] intrinsic matrices
    masks: Tensor,     # [..., C, H, W] boolean, True = ground pixel
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane:  float = 1e10,
    eps2d:      float = 0.3,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Full ground-plane depth prior pipeline from the paper.

    Steps
    -----
    1. World → camera                         (_world_to_cam)
    2. Project to 2D, get conics + radii      (persp_proj)
    3. Validity filter (depth in [near, far]) 
    4. For each camera, for each masked pixel:
       a. Unproject pixel → camera-space ray   (pinhole eq. 8)
       b. Intersect ray with Zw=0 plane        (cam2world_ground / eq. 7)
       c. Compute Euclidean depth              (calculate_depth)
    5. Return depth map D_gt for loss          (depth_supervision)

    Returns
    -------
    means2d   : [..., C, N, 2]
    conics    : [..., C, N, 3]
    depths_zc : [..., C, N]      Zc (used for tile sorting)
    radii     : [..., C, N, 2]
    D_gt      : [..., C, H, W]   ground-plane depth map (0 = not ground)
    valid     : [..., C, N]      bool mask — Gaussians that passed near/far
    """
    # ---- Step 1: world → camera ----
    means_c, covars_c = world_to_cam(means, covars, viewmats)

    # ---- Step 2: perspective projection ----
    means2d, cov2d, conics, depths_zc, radii = persp_proj(
        means_c, covars_c, Ks, width, height, eps2d
    )

    # ---- Step 3: validity filter ----
    valid = (depths_zc > near_plane) & (depths_zc < far_plane)
    radii_out = radii.clone()
    radii_out[~valid] = 0

    inside = (
        (means2d[..., 0] + radii_out[..., 0] > 0)
        & (means2d[..., 0] - radii_out[..., 0] < width)
        & (means2d[..., 1] + radii_out[..., 1] > 0)
        & (means2d[..., 1] - radii_out[..., 1] < height)
    )
    radii_out[~inside] = 0

    # ---- Step 4: ground-plane depth map ----
    # Extract camera intrinsics
    fx = Ks[..., 0, 0]   # [..., C]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]

    batch_dims = means.shape[:-2]
    C = viewmats.shape[-3]

    # Initialise output depth map with zeros
    D_gt = torch.zeros(
        batch_dims + (C, height, width),
        device=means.device, dtype=means.dtype
    )

    # Iterate over cameras (typically C is small, e.g. 30–100)
    for c in range(C):
        mask_c  = masks[..., c, :, :]           # [..., H, W]  bool
        fx_c    = fx[..., c]                     # [...]
        fy_c    = fy[..., c]
        cx_c    = cx[..., c]
        cy_c    = cy[..., c]
        vm_c    = viewmats[..., c:c+1, :, :]    # [..., 1, 4, 4]

        # All masked pixel coordinates
        ys, xs  = torch.where(mask_c[0] if mask_c.dim() > 2 else mask_c)
        if xs.numel() == 0:
            continue

        xs_f = xs.float()
        ys_f = ys.float()

        # Pinhole unprojection: (u,v) → normalised camera ray (Xc, Yc)
        # This is eq. (8) in the paper
        Xc = (xs_f - cx_c.squeeze()) / fx_c.squeeze()   # [M]
        Yc = (ys_f - cy_c.squeeze()) / fy_c.squeeze()   # [M]

        # Stack into [..., 1, M, 2]  for cam2world_ground
        rays = torch.stack([Xc, Yc], dim=-1)[None, None, :, :]  # [1, 1, M, 2]

                # Intersect rays with Zw = 0  (eq. 7)
        Xw_pts    = cam2world_ground(rays, vm_c)      # [1, 1, M, 2]
        depth_pts = calculate_depth(Xw_pts, vm_c)     # [1, 1, M]
        depth_vals = depth_pts.squeeze([0, 1])         # [M]

        # Filter 1: depth must be in valid range
        valid_depth = (depth_vals > near_plane) & (depth_vals < far_plane)

        # Filter 2: pixel coords must be inside the image
        # ys and xs come from torch.where(mask_c) so they are already
        # in-bounds for mask_c, but degenerate geometry can produce
        # world points that back-project to coordinates outside [0,H) [0,W)
        valid_bounds = (
            (ys >= 0) & (ys < height) &
            (xs >= 0) & (xs < width)
        )

        valid_pts = valid_depth & valid_bounds

        if valid_pts.any():
            D_gt[..., c, ys[valid_pts], xs[valid_pts]] = depth_vals[valid_pts]

    return means2d, conics, depths_zc, radii_out.int(), D_gt, valid


# =============================================================================
# SECTION 6 — LOSSES
# =============================================================================

def depth_supervision_loss(
    D_pred: Tensor,  # [..., C, N]   rendered depth from NeRF / GS
    D_gt:   Tensor,  # [..., C, H, W] ground-truth from ground_plane_depth_guided
    means2d: Tensor, # [..., C, N, 2] projected Gaussian centres
    width: int,
    height: int,
) -> Tensor:
    """
    Partial depth supervision loss — paper eq. (13–14):

        Ldist(r) = loss(D̂(r) − D(r))   if D(r) > 0
                   0                     if D(r) = 0

    Implementation: sample D_gt at each Gaussian's projected pixel centre,
    then compute MSE only where D_gt > 0.
    """
    # Sample D_gt at projected Gaussian positions
    # Normalise pixel coords to [-1, 1] for grid_sample
    norm_x = (means2d[..., 0] / (width  - 1)) * 2 - 1
    norm_y = (means2d[..., 1] / (height - 1)) * 2 - 1
    grid   = torch.stack([norm_x, norm_y], dim=-1)   # [..., C, N, 2]

    # D_gt: [..., C, H, W] → add channel dim → [..., C, 1, H, W]
    D_gt_4d = D_gt.unsqueeze(-3)
    # grid_sample expects [B, H, W, 2]; reshape accordingly
    orig_shape = grid.shape[:-1]
    B = math.prod(orig_shape[:-1])        # batch × C
    N = orig_shape[-1]
    D_gt_flat  = D_gt_4d.reshape(B, 1, height, width)
    grid_flat  = grid.reshape(B, 1, N, 2)

    D_sampled = F.grid_sample(
        D_gt_flat, grid_flat,
        mode='bilinear', padding_mode='zeros', align_corners=True
    ).reshape(orig_shape)  # [..., C, N]

    # Mask: only supervise where ground depth is valid
    valid_mask = D_sampled > 0.0

    if not valid_mask.any():
        return torch.tensor(0.0, device=D_pred.device, requires_grad=True)

    loss = F.mse_loss(D_pred[valid_mask], D_sampled[valid_mask])
    return loss


def color_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """L2 colour reconstruction loss — eq. (11)."""
    return F.mse_loss(rendered, target)


def total_loss(
    rendered:  Tensor,
    target:    Tensor,
    D_pred:    Tensor,
    D_gt:      Tensor,
    means2d:   Tensor,
    width:     int,
    height:    int,
    lambda_dist: float = 2.3,  # paper sets λ = 2.3
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Full training loss from paper eq. (14):

        L = Lcolor + λ · Ldist
    """
    Lcolor = color_loss(rendered, target)
    Ldist  = depth_supervision_loss(D_pred, D_gt, means2d, width, height)
    L      = Lcolor + lambda_dist * Ldist
    return L, Lcolor, Ldist


# =============================================================================
# SECTION 7 — SCALE ALIGNMENT  (ZoeDepth comparison, eqs. 15–16)
# =============================================================================

def align_depth_scale(
    D_pre:    Tensor,    # [...] estimated depth (ZoeDepth output)
    D_sparse: Tensor,    # [...] sparse GT from point-cloud projection
    weights:  Optional[Tensor] = None,  # [...] reliability weights w ∈ [0,1]
) -> Tuple[float, float, Tensor]:
    """
    Find scale s and shift t such that  s·D_pre + t ≈ D_sparse
    using weighted least-squares (paper eqs. 15–16).

    Only uses pixels where D_sparse > 0 (i.e. where the sparse cloud projects).

    Returns: (s, t, D_reg) where D_reg = s·D_pre + t
    """
    valid = D_sparse > 0
    x = D_pre[valid].float()
    y = D_sparse[valid].float()
    w = weights[valid].float() if weights is not None else torch.ones_like(x)

    # Weighted least-squares:  min_s,t  Σ w·(y − s·x − t)²
    # Closed form:
    #   s = (Σw·Σwxy − Σwx·Σwy) / (Σw·Σwx² − (Σwx)²)
    #   t = (Σwy − s·Σwx) / Σw
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

    D_reg = s * D_pre + t
    return s, t, D_reg


# =============================================================================
# SECTION 8 — GAUSSIAN MODEL  (learnable parameters)
# =============================================================================

class GaussianModel(nn.Module):
    """
    Learnable 3D Gaussian scene representation.

    Each Gaussian has:
      means    — 3D world-space centre     (Xw, Yw, Zw)
      quats    — rotation quaternion       (w, x, y, z)
      scales   — log scale (3 axes)
      opacities— pre-sigmoid opacity
      sh_coeffs— spherical harmonic colour coefficients
    """

    def __init__(self, N: int, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = sh_degree
        K = (sh_degree + 1) ** 2  # number of SH bands

        self.means     = nn.Parameter(torch.randn(N, 3))
        self.quats     = nn.Parameter(F.normalize(torch.randn(N, 4), dim=-1))
        self.log_scales= nn.Parameter(torch.zeros(N, 3))
        self.logit_opacities = nn.Parameter(torch.zeros(N))
        self.sh_coeffs = nn.Parameter(torch.zeros(N, K, 3))

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.logit_opacities)

    @property
    def covars(self) -> Tensor:
        return quat_scale_to_covar(self.quats, self.scales)

    def get_colors(self, dirs: Tensor) -> Tensor:
        """
        Evaluate SH colour for view directions dirs [..., 3].
        Returns [..., N, 3] RGB colours in [0, 1].
        """
        from torch import Tensor as T
        dirs = F.normalize(dirs, dim=-1)
        # Simple DC term for now; extend with full SH eval if needed
        colors = self.sh_coeffs[..., 0, :] * 0.28209479 + 0.5
        return colors.clamp(0, 1)


# =============================================================================
# SECTION 9 — FULL FORWARD PASS
# =============================================================================

def render(
    model:     GaussianModel,
    viewmats:  Tensor,   # [C, 4, 4]
    Ks:        Tensor,   # [C, 3, 3]
    width:     int,
    height:    int,
    masks:     Tensor,   # [C, H, W]  ground segmentation
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane:  float  = 1e10,
    lambda_dist: float = 2.3,
    target_image: Optional[Tensor] = None,  # [C, H, W, 3]
) -> dict:
    """
    Full forward render pass with ground-plane depth supervision.

    Returns a dict with:
      means2d, conics, depths, radii  — projection outputs
      D_gt                            — ground-plane depth maps
      isect_offsets, flatten_ids      — tile rasterisation tables
      loss (if target_image given)    — total training loss
    """
    C = viewmats.shape[0]
    N = model.means.shape[0]
    tile_w = math.ceil(width  / tile_size)
    tile_h = math.ceil(height / tile_size)

    # ---- Forward projection + depth maps ----
    means2d, conics, depths_zc, radii, D_gt, valid = ground_plane_depth_guided(
        model.means, model.covars,
        viewmats, Ks, masks,
        width, height,
        near_plane=near_plane, far_plane=far_plane,
    )

    # ---- Tile intersection ----
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths_zc,
        tile_size, tile_w, tile_h,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_w, tile_h)

    out = dict(
        means2d=means2d,
        conics=conics,
        depths_zc=depths_zc,
        radii=radii,
        D_gt=D_gt,
        valid=valid,
        isect_ids=isect_ids,
        flatten_ids=flatten_ids,
        isect_offsets=isect_offsets,
    )

    if target_image is not None:
        # Placeholder rendered image — in practice this comes from
        # _rasterize_to_pixels using the tile tables above.
        rendered = torch.zeros_like(target_image)

        L, Lcolor, Ldist = total_loss(
            rendered, target_image,
            depths_zc, D_gt, means2d,
            width, height,
            lambda_dist=lambda_dist,
        )
        out.update(loss=L, Lcolor=Lcolor, Ldist=Ldist)

    return out


# =============================================================================
# SECTION 10 — TRAINING LOOP SKETCH
# =============================================================================

def train(
    model: GaussianModel,
    viewmats_list: list,   # list of [C, 4, 4] tensors per training step
    Ks_list: list,
    masks_list: list,
    targets_list: list,    # list of [C, H, W, 3] ground-truth images
    width: int,
    height: int,
    n_iters: int = 30_000,
    lr: float = 1e-3,
    lambda_dist: float = 2.3,
):
    """
    Minimal training loop following the paper:
      - 30 000 iterations
      - Adam optimiser
      - Loss = Lcolor + 2.3 · Ldist
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_iters):
        idx = step % len(viewmats_list)
        viewmats = viewmats_list[idx]
        Ks       = Ks_list[idx]
        masks    = masks_list[idx]
        target   = targets_list[idx]

        optimiser.zero_grad()

        out = render(
            model, viewmats, Ks,
            width, height, masks,
            target_image=target,
            lambda_dist=lambda_dist,
        )

        loss = out["loss"]
        loss.backward()
        optimiser.step()

        if step % 1000 == 0:
            print(
                f"step {step:5d}  "
                f"loss={loss.item():.4f}  "
                f"Lcolor={out['Lcolor'].item():.4f}  "
                f"Ldist={out['Ldist'].item():.4f}"
            )

    return model


# =============================================================================
# QUICK SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    N, C, H, W = 64, 2, 64, 64

    model = GaussianModel(N, sh_degree=1).to(device)

    # Identity cameras looking forward along +Z
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = -5.0  # move cameras back 5 units

    Ks = torch.tensor([[
        [50.0,  0.0, W/2],
        [ 0.0, 50.0, H/2],
        [ 0.0,  0.0,  1.0],
    ]], device=device).repeat(C, 1, 1)

    # Mark lower half of each camera image as ground
    masks = torch.zeros(C, H, W, dtype=torch.bool, device=device)
    masks[:, H//2:, :] = True

    out = render(model, viewmats, Ks, W, H, masks)

    print("means2d shape  :", out["means2d"].shape)
    print("conics shape   :", out["conics"].shape)
    print("D_gt shape     :", out["D_gt"].shape)
    print("isect_offsets  :", out["isect_offsets"].shape)
    print("n_isects       :", out["flatten_ids"].shape[0])
    print("Smoke test passed.")