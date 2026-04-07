import math
import struct
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




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


