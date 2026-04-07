import math
import struct
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils_prior import quat_scale_to_covar, world_to_cam, quat_to_rotmat, persp_proj


def align_depth_scale(
    D_pred: Tensor, 
    D_sparse: Tensor,
    weight: Optional[Tensor] = None,
    ) -> Tensor:
    """
    D_pre from DA3 Output Model
    D_gt from Alpha Blending Ground Truth Depth
    Weights the reliabitity of each point in the D_sparse, weights for each point
    """
    x = D_pred.float()
    y = D_sparse.float()
    W = weight.float() if weight is not None else torch.ones_like(x)

    # Weighted Least Squares
    sw   = w.sum()
    swx  = (w * x).sum()
    swy  = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()
    denom = sw * swxx - swx * swx + 1e-8  # Avoid division by zero
    a = (sw * swxy - swx * swy) / denom
    b = (swxx * swy - swx * swxy) / denom
    D_aligned = a * D_pred + b
    return D_aligned

def ground_plane_depth_guided(
    means:Tensor,    # [..., N, 3]
    covars:Tensor,   # [..., N, 3, 3]
    viewmats:Tensor, # [..., C, 4, 4]
    Ks:Tensor,      # [..., C, 3, 3]
    masks:Tensor,    # [..., C, H, W]
    near_plane:float = 0.01,
    far_plane:float = 1e10,
    eps2d:float = 0.3,
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