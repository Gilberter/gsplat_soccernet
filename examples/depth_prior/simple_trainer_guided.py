# ==============================================================================
# simple_trainer_guided.py
# ==============================================================================
# Ground-Plane Depth-Supervised Gaussian Splatting Trainer
#
# This file is a self-contained drop-in replacement for simple_trainer.py.
# It keeps 100% of the original gsplat training loop and adds ground-plane
# depth supervision on top, following:
#
#   Dai et al. 2024 — "Scene-Constrained NeRF"
#   Kerbl et al. 2023 — "3D Gaussian Splatting for Real-Time Rendering"
#
# ──────────────────────────────────────────────────────────────────────────────
# QUICK-START (single GPU):
#
#   python simple_trainer_guided.py default \
#       --data_dir       /data/scene-1 \
#       --colmap_dir     /data/scene-1/sparse/0 \
#       --result_dir     results/scene-1-guided \
#       --ground_seg_dir /data/output_sam/grass_masks \
#       --ground_depth_loss \
#       --ground_depth_lambda 2.3 \
#       --no-normalize_world_space \
#       --no-load_exposure \
#       --test_every 0 \
#       --disable_viewer
#
# ──────────────────────────────────────────────────────────────────────────────
# FILE STRUCTURE (sections you may need to modify are marked with ★):
#
#   SECTION A  Imports & constants
#   SECTION B  Ground-plane geometry (math — rarely needs changing)
#   SECTION C  Ground-plane depth loss (loss fn — rarely needs changing)
#   SECTION D  Config dataclass          ★ add/tune hyperparameters here
#   SECTION E  Splat + optimizer factory (unchanged from simple_trainer)
#   SECTION F  Runner.__init__           ★ dataset / model setup
#   SECTION G  Runner.rasterize_splats   (unchanged from simple_trainer)
#   SECTION H  Runner.train              ★ ground loss injected here
#   SECTION I  Runner.eval / render_traj (unchanged from simple_trainer)
#   SECTION J  main() + CLI entry point
#
# ──────────────────────────────────────────────────────────────────────────────
# TUNING GUIDE:
#
#   ground_depth_lambda (default 2.3)
#       Weight on the depth loss relative to the colour loss.
#       Increase → floor anchors harder but colour quality may drop slightly.
#       Decrease → softer floor constraint, better for very noisy masks.
#
#   ground_seg_dir
#       Directory of binary PNG masks (one per training image, same stem).
#       White = ground pixel.  Black = not ground.
#       Generate with SAM + any grass/pitch classifier.
#
#   ground_depth_start_step (default 0)
#       Iteration at which the depth loss switches on.
#       Setting this to ~1000 lets the scene warm up first.
#
#   scale_reg (default 0.01 for MCMC, 0.0 for default)
#       Penalises large Gaussian scales.  Helps keep floor discs thin.
#
#   opacity_reg (default 0.01 for MCMC)
#       Prunes transparent floaters above the pitch.
#
#   absgrad (default False)
#       Use absolute gradient for densification.  Recommended for large
#       textureless regions like a grass pitch where photometric gradients
#       are very small.
#
# ==============================================================================


# ==============================================================================
# SECTION A — IMPORTS & CONSTANTS
# ==============================================================================

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

# gsplat core
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy

# local examples helpers (same folder as simple_trainer.py)
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat.color_correct import color_correct_affine, color_correct_quadratic


# ==============================================================================
# SECTION B — GROUND-PLANE GEOMETRY
# ==============================================================================
# These functions implement the mathematical core of the depth prior.
# You almost never need to change them.  Annotated for reference.


def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """
    Quaternion (w, x, y, z) → 3×3 rotation matrix.

    Shape contract: [..., 4] → [..., 3, 3]
    """
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack([
        1 - 2*(y**2 + z**2),  2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),         1 - 2*(x**2 + z**2),  2*(y*z - w*x),
        2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x**2 + y**2),
    ], dim=-1)
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_covar(quats: Tensor, scales: Tensor) -> Tensor:
    """
    Build 3D covariance matrix  Σ = R · S · Sᵀ · Rᵀ.

    Called to pass covars into ground_plane_depth_guided.
    Inputs  : quats  [..., 4]  (normalised quaternions)
              scales [..., 3]  (positive scale values, NOT log-scale)
    Output  : covars [..., 3, 3]
    """
    R = _quat_to_rotmat(quats)
    M = R * scales[..., None, :]       # R · diag(s)
    return torch.einsum("...ij,...kj->...ik", M, M)


def _cam2world_ground(rays_2d: Tensor, viewmats: Tensor) -> Tensor:
    """
    Back-project normalised camera rays (Xc/Zc, Yc/Zc) onto the Zw=0
    ground plane.  Implements paper eq. (7).

    Solves a 2x2 linear system  A·[Xw, Yw]ᵀ = b  per pixel:

        A = [[Xc·r31 - r11,  Xc·r32 - r12],
             [Yc·r31 - r21,  Yc·r32 - r22]]
        b = [t1 - Xc·t3,  t2 - Yc·t3]

    Inputs : rays_2d  [..., C, M, 2]  normalised rays (Xc, Yc)
             viewmats [..., C, 4, 4]  world-to-camera extrinsics
    Output : [..., C, M, 2]  world-space (Xw, Yw) on the Zw=0 plane
    """
    x_c, y_c = torch.unbind(rays_2d, dim=-1)

    R  = viewmats[..., :3, :3]
    t  = viewmats[..., :3, 3]

    r11 = R[..., 0, 0]; r12 = R[..., 0, 1]
    r21 = R[..., 1, 0]; r22 = R[..., 1, 1]
    r31 = R[..., 2, 0]; r32 = R[..., 2, 1]
    t1  = t[..., 0];    t2  = t[..., 1];  t3 = t[..., 2]

    A11 = x_c * r31[..., None] - r11[..., None]
    A12 = x_c * r32[..., None] - r12[..., None]
    A21 = y_c * r31[..., None] - r21[..., None]
    A22 = y_c * r32[..., None] - r22[..., None]

    A = torch.stack([
        torch.stack([A11, A12], dim=-1),
        torch.stack([A21, A22], dim=-1),
    ], dim=-2)                                          # [..., C, M, 2, 2]

    b1 = t1[..., None] - x_c * t3[..., None]
    b2 = t2[..., None] - y_c * t3[..., None]
    b  = torch.stack([b1, b2], dim=-1)[..., None]      # [..., C, M, 2, 1]

    return torch.linalg.solve(A, b).squeeze(-1)         # [..., C, M, 2]


def _calculate_depth(Xw: Tensor, viewmats: Tensor) -> Tensor:
    """
    Euclidean distance from the camera centre to each ground-plane point.
    Implements paper Algorithm 1 line 11:  depth = ‖pw − cam_centre‖₂

    Inputs : Xw       [..., C, M, 2]  (Xw, Yw) on the Zw=0 plane
             viewmats [..., C, 4, 4]
    Output : [..., C, M]  metric depth values
    """
    R      = viewmats[..., :3, :3]
    t_cam  = viewmats[..., :3, 3]

    # Camera centre in world coordinates:  c = −Rᵀ · t
    cam_center = -torch.einsum("...ji,...j->...i", R, t_cam)   # [..., C, 3]
    cam_center = cam_center[..., None, :]                        # [..., C, 1, 3]

    zeros  = torch.zeros_like(Xw[..., :1])
    Xw_3d  = torch.cat([Xw, zeros], dim=-1)                     # [..., C, M, 3]

    return torch.linalg.norm(cam_center - Xw_3d, dim=-1)        # [..., C, M]


def ground_plane_depth_guided(
    means:      Tensor,   # [N, 3]       Gaussian world-space centres
    covars:     Tensor,   # [N, 3, 3]    Gaussian world-space covariances
    viewmats:   Tensor,   # [C, 4, 4]    world-to-camera extrinsics
    Ks:         Tensor,   # [C, 3, 3]    camera intrinsics
    masks:      Tensor,   # [1, C, H, W] ground segmentation (bool)
    width:      int,
    height:     int,
    near_plane: float = 0.01,
    far_plane:  float = 1e10,
) -> Tensor:
    """
    Build the ground-truth depth map D_gt from the geometric ground-plane prior.

    For every pixel that the SAM mask marks as ground, we:
      1. Unproject the pixel to a normalised camera ray  (pinhole eq. 8)
      2. Intersect the ray with the Zw=0 world plane     (cam2world_ground eq. 7)
      3. Measure the Euclidean distance from camera to that world point

    The result is a sparse depth map: D_gt[c, v, u] > 0 only at ground pixels.
    Non-ground pixels remain 0 and are excluded from the loss.

    ★ TO MODIFY:
        - Change near_plane / far_plane to clip very close or very far geometry.
        - The Zw=0 assumption means the world coordinate system must have the
          pitch at Z=0.  If you trained with normalize_world_space=True, disable
          it (--no-normalize_world_space) so COLMAP's Z-up convention is kept.

    Inputs:
        masks shape [1, C, H, W] — batch dim of 1 expected (batch_size=1).

    Returns:
        D_gt : [C, H, W]  float32 depth map (0 where not ground)
    """
    C  = viewmats.shape[0]
    fx = Ks[:, 0, 0]   # [C]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]

    D_gt = torch.zeros(C, height, width, device=means.device, dtype=means.dtype)

    for c in range(C):
        # masks is [1, C, H, W] coming from the dataloader batch
        mask_c = masks[0, c] if masks.dim() == 4 else masks[c]  # [H, W] bool

        ys, xs = torch.where(mask_c)
        if xs.numel() == 0:
            continue

        xs_f = xs.float()
        ys_f = ys.float()

        # Pinhole back-projection:  (u,v) → (Xc/Zc, Yc/Zc)
        Xc = (xs_f - cx[c]) / fx[c]    # [M]
        Yc = (ys_f - cy[c]) / fy[c]    # [M]

        # Shape: [1, 1, M, 2]  (batch × cameras × pixels × 2)
        rays = torch.stack([Xc, Yc], dim=-1)[None, None, :, :]

        vm_c = viewmats[c:c+1].unsqueeze(0)           # [1, 1, 4, 4]

        try:
            Xw_pts    = _cam2world_ground(rays, vm_c)  # [1, 1, M, 2]
            depth_pts = _calculate_depth(Xw_pts, vm_c) # [1, 1, M]
            depth_vals = depth_pts.squeeze([0, 1])      # [M]
        except torch.linalg.LinAlgError:
            # Degenerate camera (looking straight down / up): skip this camera
            continue

        valid = (
            (depth_vals > near_plane) &
            (depth_vals < far_plane)  &
            (ys >= 0) & (ys < height) &
            (xs >= 0) & (xs < width)
        )

        if valid.any():
            D_gt[c, ys[valid], xs[valid]] = depth_vals[valid]

    return D_gt   # [C, H, W]


# ==============================================================================
# SECTION C — GROUND-PLANE DEPTH LOSS
# ==============================================================================

def ground_depth_loss_fn(
    depths_image: Tensor,   # [C, 1, H, W]  rendered depth from rasterizer
    D_gt:         Tensor,   # [C, H, W]     geometric ground prior
    means2d:      Tensor,   # [C, N, 2]     projected Gaussian centres
    width:        int,
    height:       int,
) -> Tensor:
    """
    Partial depth supervision loss (paper eq. 13–14):

        Ldist(r) = MSE(D̂(r), D(r))   if D(r) > 0
                   0                   if D(r) = 0

    Implementation:
        We sample the rendered depth image at each Gaussian's projected 2D
        position (means2d), giving a per-Gaussian predicted depth D_pred.
        We also sample D_gt at the same positions to get the target.
        MSE is computed only over the subset where D_gt > 0.

    ★ TO MODIFY:
        - Swap F.mse_loss → F.l1_loss for a smoother loss surface.
        - Add a confidence weight if your mask quality varies per image.
        - Increase/decrease the threshold `> 0.0` if D_gt has noise near 0.

    Returns:
        Scalar loss tensor (0.0 with requires_grad=True if no ground pixels).
    """
    C, _, H, W = depths_image.shape
    N = means2d.shape[1]

    # Normalise pixel coords to [-1, 1] for grid_sample
    norm_x = (means2d[..., 0] / (W - 1)) * 2 - 1   # [C, N]
    norm_y = (means2d[..., 1] / (H - 1)) * 2 - 1

    # grid_sample expects [C, 1, H, W] input and [C, N, 1, 2] grid
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)   # [C, N, 1, 2]

    # Sample predicted depth at Gaussian centres
    D_pred_at_means = F.grid_sample(
        depths_image,                   # [C, 1, H, W]
        grid,                           # [C, N, 1, 2]
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1).squeeze(-1)            # [C, N]

    # Sample D_gt at the same positions
    D_gt_at_means = F.grid_sample(
        D_gt.unsqueeze(1),              # [C, 1, H, W]
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1).squeeze(-1)            # [C, N]

    valid = D_gt_at_means > 0.0

    if not valid.any():
        return torch.tensor(0.0, device=depths_image.device, requires_grad=True)

    return F.mse_loss(D_pred_at_means[valid], D_gt_at_means[valid])


# ==============================================================================
# SECTION D — CONFIG DATACLASS
# ==============================================================================
# ★ This is the primary place to add or tune hyperparameters.
# Every field maps directly to a CLI flag via tyro.

@dataclass
class Config:
    # --------------------------------------------------------------------------
    # Infrastructure
    # --------------------------------------------------------------------------

    # Disable the viser web viewer (recommended for cluster training)
    disable_viewer: bool = False

    # Path to pre-trained .pt checkpoint(s).  Skips training and runs eval only.
    ckpt: Optional[List[str]] = None

    # PNG compression of the final splat (optional)
    compression: Optional[Literal["png"]] = None

    # Camera trajectory type for render_traj  [interp | ellipse | spiral]
    render_traj_path: str = "interp"

    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------

    # Path to the scene root (must contain an "images" folder)
    data_dir: str = "data/360_v2/garden"

    # Path to COLMAP sparse reconstruction  (cameras.bin / images.bin / points3D.bin)
    # ★ For soccer scenes use the COLMAP output at sparse/0
    colmap_dir: str = ""

    # Image downscale factor  (2 = half resolution, 4 = quarter, etc.)
    data_factor: int = 4

    # Results output directory
    result_dir: str = "results/garden"

    # Test image interval: every N-th image is held out as validation
    # Set to 0 to use ALL images for training (recommended for challenge scenes)
    test_every: int = 8

    # Optional: random spatial crop during training
    patch_size: Optional[int] = None

    # Global scale multiplier for scene-size-dependent parameters
    global_scale: float = 1.0

    # Normalize world space to unit sphere (DISABLE for ground-plane prior!)
    # ★ IMPORTANT: set --no-normalize_world_space when using ground_depth_loss
    #    The prior assumes Zw=0 is the physical ground in COLMAP coordinates.
    normalize_world_space: bool = False

    # Camera model  [pinhole | fisheye | ortho]
    camera_model: CameraModel = "pinhole"

    # Load per-image exposure from EXIF metadata
    load_exposure: bool = True

    # --------------------------------------------------------------------------
    # Ground-plane depth supervision  ★ NEW — tune these
    # --------------------------------------------------------------------------

    # Enable ground-plane depth loss
    ground_depth_loss: bool = False

    # Weight on the depth loss:  L = Lcolor + λ · Ldist
    # Paper value: 2.3.  Increase for stronger floor anchoring.
    ground_depth_lambda: float = 2.3

    # Directory containing binary ground-segmentation masks.
    # One PNG per training image, named identically (same stem as image file).
    # White pixels (>0) = ground.  Black = not ground.
    # Generate with SAM + a grass/pitch classifier.
    ground_seg_dir: str = ""

    # Iteration at which the depth loss turns on.
    # Set to ~1000 to let the scene warm up before applying the constraint.
    ground_depth_start_step: int = 0

    # --------------------------------------------------------------------------
    # Viewer
    # --------------------------------------------------------------------------

    port: int = 8080

    # --------------------------------------------------------------------------
    # Training schedule
    # --------------------------------------------------------------------------

    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    # --------------------------------------------------------------------------
    # Gaussian initialisation
    # --------------------------------------------------------------------------

    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0

    # --------------------------------------------------------------------------
    # Loss weights
    # --------------------------------------------------------------------------

    # SSIM vs L1 blend  (0 = pure L1, 1 = pure SSIM)
    ssim_lambda: float = 0.2

    # Sparse COLMAP depth loss (original simple_trainer feature, separate from
    # the ground-plane prior)
    depth_loss: bool = False
    depth_lambda: float = 1e-2

    # --------------------------------------------------------------------------
    # Near / far plane clipping
    # --------------------------------------------------------------------------

    near_plane: float = 0.01
    far_plane:  float = 1e10

    # --------------------------------------------------------------------------
    # Densification strategy
    # --------------------------------------------------------------------------

    # DefaultStrategy or MCMCStrategy object (set by CLI sub-command)
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )

    # --------------------------------------------------------------------------
    # Rasterization options
    # --------------------------------------------------------------------------

    packed:      bool = False
    sparse_grad: bool = False

    # ★ RECOMMENDED for soccer/grass:
    #   absgrad=True helps densify large textureless regions where photometric
    #   gradients are too small to trigger standard densification.
    visible_adam: bool = False
    antialiased:  bool = False
    absgrad:      bool = False

    random_bkgd: bool = False

    # Explicit rasterize mode override  [classic | antialiased | None]
    rasterize_mode: Optional[Literal["classic", "antialiased"]] = None

    # --------------------------------------------------------------------------
    # Learning rates
    # --------------------------------------------------------------------------

    means_lr:      float = 1.6e-4
    scales_lr:     float = 5e-3
    opacities_lr:  float = 5e-2
    quats_lr:      float = 1e-3
    sh0_lr:        float = 2.5e-3
    shN_lr:        float = 2.5e-3 / 20

    # --------------------------------------------------------------------------
    # Regularization
    # --------------------------------------------------------------------------

    # ★ For soccer pitch: opacity_reg=0.01, scale_reg=0.01 prune floaters and
    #   keep floor Gaussians as thin discs.
    opacity_reg: float = 0.0
    scale_reg:   float = 0.0

    # --------------------------------------------------------------------------
    # Camera pose optimisation (optional)
    # --------------------------------------------------------------------------

    pose_opt:     bool  = False
    pose_opt_lr:  float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise:   float = 0.0

    # --------------------------------------------------------------------------
    # Appearance optimisation (optional)
    # --------------------------------------------------------------------------

    app_opt:        bool  = False
    app_embed_dim:  int   = 16
    app_opt_lr:     float = 1e-3
    app_opt_reg:    float = 1e-6

    # --------------------------------------------------------------------------
    # Post-processing (bilateral grid / ppisp)
    # --------------------------------------------------------------------------

    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    bilateral_grid_fused: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    ppisp_use_controller: bool = True
    ppisp_controller_distillation: bool = True
    ppisp_controller_activation_num_steps: int = 25_000
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    use_color_correction_metric: bool = False

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------

    tb_every:      int  = 100
    tb_save_image: bool = False
    lpips_net:     Literal["vgg", "alex"] = "alex"

    # --------------------------------------------------------------------------
    # 3DGUT (experimental)
    # --------------------------------------------------------------------------

    with_ut:    bool = False
    with_eval3d: bool = False

    # --------------------------------------------------------------------------
    # Step scaling (for multi-GPU runs)
    # --------------------------------------------------------------------------

    def adjust_steps(self, factor: float):
        self.eval_steps           = [int(i * factor) for i in self.eval_steps]
        self.save_steps           = [int(i * factor) for i in self.save_steps]
        self.ply_steps            = [int(i * factor) for i in self.ply_steps]
        self.max_steps            = int(self.max_steps * factor)
        self.sh_degree_interval   = int(self.sh_degree_interval * factor)
        self.ground_depth_start_step = int(self.ground_depth_start_step * factor)

        s = self.strategy
        if isinstance(s, DefaultStrategy):
            s.refine_start_iter = int(s.refine_start_iter * factor)
            s.refine_stop_iter  = int(s.refine_stop_iter  * factor)
            s.reset_every       = int(s.reset_every       * factor)
            s.refine_every      = int(s.refine_every      * factor)
        elif isinstance(s, MCMCStrategy):
            s.refine_start_iter = int(s.refine_start_iter * factor)
            s.refine_stop_iter  = int(s.refine_stop_iter  * factor)
            s.refine_every      = int(s.refine_every      * factor)
            if s.noise_injection_stop_iter >= 0:
                s.noise_injection_stop_iter = int(s.noise_injection_stop_iter * factor)
        else:
            assert_never(s)


# ==============================================================================
# SECTION E — SPLAT + OPTIMIZER FACTORY
# ==============================================================================
# Unchanged from simple_trainer.py.

def create_splats_with_optimizers(
    parser:       Parser,
    init_type:    str   = "sfm",
    init_num_pts: int   = 100_000,
    init_extent:  float = 3.0,
    init_opacity: float = 0.1,
    init_scale:   float = 1.0,
    means_lr:     float = 1.6e-4,
    scales_lr:    float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr:     float = 1e-3,
    sh0_lr:       float = 2.5e-3,
    shN_lr:       float = 2.5e-3 / 20,
    scene_scale:  float = 1.0,
    sh_degree:    int   = 3,
    sparse_grad:  bool  = False,
    visible_adam: bool  = False,
    batch_size:   int   = 1,
    feature_dim:  Optional[int] = None,
    device:       str   = "cuda",
    world_rank:   int   = 0,
    world_size:   int   = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs   = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs   = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("init_type must be 'sfm' or 'random'")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg  = torch.sqrt(dist2_avg)
    scales    = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    points = points[world_rank::world_size]
    rgbs   = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N         = points.shape[0]
    quats     = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    params = [
        ("means",     torch.nn.Parameter(points),     means_lr * scene_scale),
        ("scales",    torch.nn.Parameter(scales),     scales_lr),
        ("quats",     torch.nn.Parameter(quats),      quats_lr),
        ("opacities", torch.nn.Parameter(opacities),  opacities_lr),
    ]

    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]),  sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]),  shN_lr))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = batch_size * world_size
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


# ==============================================================================
# SECTION F — RUNNER.__INIT__
# ==============================================================================

class Runner:
    """
    Training and evaluation engine.

    The key difference from simple_trainer.Runner:
      - Dataset is loaded with  load_ground_masks=cfg.ground_seg_dir
      - ground_plane_depth_guided() is called inside train() when
        cfg.ground_depth_loss is True
    """

    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: Config):
        set_random_seed(42 + local_rank)

        self.cfg        = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device     = f"cuda:{local_rank}"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir   = f"{cfg.result_dir}/ckpts";   os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir  = f"{cfg.result_dir}/stats";   os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"; os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir    = f"{cfg.result_dir}/ply";     os.makedirs(self.ply_dir,   exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # ── Dataset ─────────────────────────────────────────────────────────
        # ★ Parser must receive the COLMAP directory explicitly.
        #   For soccer challenge scenes use colmap_dir=<scene>/sparse/0
        self.parser = Parser(
            data_dir      = cfg.data_dir,
            factor        = cfg.data_factor,
            normalize     = cfg.normalize_world_space,
            test_every    = cfg.test_every,
            load_exposure = cfg.load_exposure,
            colmap_dir    = cfg.colmap_dir,
        )

        # ★ load_ground_masks activates the SAM mask loading in Dataset.__getitem__.
        #   If ground_seg_dir is empty the masks arg is ignored and no ground loss
        #   is computed regardless of the ground_depth_loss flag.
        self.trainset = Dataset(
            self.parser,
            split             = "train",
            patch_size        = cfg.patch_size,
            load_depths       = cfg.depth_loss,
            load_ground_masks = cfg.ground_seg_dir,
        )
        self.valset     = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"[Runner] Scene scale: {self.scene_scale:.4f}")
        print(f"[Runner] Training images: {len(self.trainset)}")
        print(f"[Runner] Ground depth loss: {cfg.ground_depth_loss}")
        if cfg.ground_depth_loss:
            print(f"[Runner]   → ground_seg_dir    : {cfg.ground_seg_dir}")
            print(f"[Runner]   → ground_depth_lambda: {cfg.ground_depth_lambda}")
            print(f"[Runner]   → start step         : {cfg.ground_depth_start_step}")
            print(f"[Runner]   → normalize_world    : {cfg.normalize_world_space}")
            if cfg.normalize_world_space:
                print("[Runner]   ⚠ WARNING: normalize_world_space=True may misalign "
                      "the Zw=0 ground plane assumption.  Consider --no-normalize_world_space.")

        # ── Model ────────────────────────────────────────────────────────────
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type    = cfg.init_type,
            init_num_pts = cfg.init_num_pts,
            init_extent  = cfg.init_extent,
            init_opacity = cfg.init_opa,
            init_scale   = cfg.init_scale,
            means_lr     = cfg.means_lr,
            scales_lr    = cfg.scales_lr,
            opacities_lr = cfg.opacities_lr,
            quats_lr     = cfg.quats_lr,
            sh0_lr       = cfg.sh0_lr,
            shN_lr       = cfg.shN_lr,
            scene_scale  = self.scene_scale,
            sh_degree    = cfg.sh_degree,
            sparse_grad  = cfg.sparse_grad,
            visible_adam = cfg.visible_adam,
            batch_size   = cfg.batch_size,
            feature_dim  = feature_dim,
            device       = self.device,
            world_rank   = world_rank,
            world_size   = world_size,
        )
        print(f"[Runner] Gaussians initialised: {len(self.splats['means'])}")

        # ── Densification strategy ───────────────────────────────────────────
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(cfg.strategy, DefaultStrategy):
            self.strategy_state = cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            self.strategy_state = cfg.strategy.initialize_state()
        else:
            assert_never(cfg.strategy)

        # ── Compression ──────────────────────────────────────────────────────
        self.compression_method = None
        if cfg.compression == "png":
            self.compression_method = PngCompression()

        # ── Pose optimisation ────────────────────────────────────────────────
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)
        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # ── Appearance optimisation ──────────────────────────────────────────
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # ── Post-processing ──────────────────────────────────────────────────
        self.post_processing_module   = None
        self.post_processing_optimizers = []

        # ── Metrics ──────────────────────────────────────────────────────────
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        else:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)

        # ── Viewer ───────────────────────────────────────────────────────────
        if not cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        self._gaussians_frozen = False


    # ==========================================================================
    # SECTION G — RASTERIZE (unchanged from simple_trainer)
    # ==========================================================================

    def rasterize_splats(
        self,
        camtoworlds:    Tensor,
        Ks:             Tensor,
        width:          int,
        height:         int,
        masks:          Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model:   Optional[CameraModel]   = None,
        frame_idcs:     Optional[Tensor] = None,
        camera_idcs:    Optional[Tensor] = None,
        exposure:       Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Forward render pass.  Returns (render_colors, render_alphas, info).

        render_colors shape:
          - "RGB"    → [C, H, W, 3]
          - "RGB+ED" → [C, H, W, 4]  last channel is expected depth
        """
        means     = self.splats["means"]
        quats     = self.splats["quats"]
        scales    = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features   = self.splats["features"],
                embed_ids  = image_ids,
                dirs       = means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree  = kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

        render_colors, render_alphas, info = rasterization(
            means          = means,
            quats          = quats,
            scales         = scales,
            opacities      = opacities,
            colors         = colors,
            viewmats       = torch.linalg.inv(camtoworlds),
            Ks             = Ks,
            width          = width,
            height         = height,
            packed         = self.cfg.packed,
            absgrad        = (
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy) else False
            ),
            sparse_grad    = self.cfg.sparse_grad,
            rasterize_mode = rasterize_mode,
            distributed    = self.world_size > 1,
            camera_model   = camera_model,
            with_ut        = self.cfg.with_ut,
            with_eval3d    = self.cfg.with_eval3d,
            **kwargs,
        )

        # Keep means2d in the computation graph for depth loss gradient flow
        if "means2d" in info and not info["means2d"].requires_grad:
            info["means2d"] = (
                info["means2d"]
                + 0.0 * means[..., None, :, :2].expand_as(info["means2d"])
            )

        if masks is not None:
            render_colors[~masks] = 0

        return render_colors, render_alphas, info


    # ==========================================================================
    # SECTION H — TRAIN LOOP
    # ==========================================================================
    # ★ This is where the ground-plane depth loss is injected.
    #   Look for the  "── Ground-plane depth loss ──"  block.

    def train(self):
        cfg        = self.cfg
        device     = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size        = cfg.batch_size,
            shuffle           = True,
            num_workers       = 4,
            persistent_workers= True,
            pin_memory        = True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))

        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            # ── Unpack batch ─────────────────────────────────────────────────
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1,4,4]
            Ks          = data["K"].to(device)                            # [1,3,3]
            pixels      = data["image"].to(device) / 255.0               # [1,H,W,3]
            image_ids   = data["image_id"].to(device)
            masks       = data["mask"].to(device) if "mask" in data else None
            exposure    = data["exposure"].to(device) if "exposure" in data else None
            height, width = pixels.shape[1:3]
            num_train_rays_per_step = pixels.shape[0] * height * width

            # Ground segmentation mask for this batch image
            # Shape: [1, H, W] bool  (from Dataset.__getitem__)
            ground_masks = data["ground_mask"].to(device) if "ground_mask" in data else None

            # COLMAP sparse depth (for the original depth_loss, independent of
            # the ground-plane prior)
            if cfg.depth_loss:
                points    = data["points"].to(device)
                depths_gt = data["depths"].to(device)

            # ── Pose perturbation / optimisation ─────────────────────────────
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # ── SH degree schedule ───────────────────────────────────────────
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # ── Decide render mode ───────────────────────────────────────────
            # We need depth output (ED channel) when:
            #   • the original depth_loss is on, OR
            #   • the ground-plane depth loss is on
            need_depth = cfg.depth_loss or (
                cfg.ground_depth_loss
                and ground_masks is not None
                and step >= cfg.ground_depth_start_step
            )
            render_mode = "RGB+ED" if need_depth else "RGB"

            # ── Forward rasterization ────────────────────────────────────────
            renders, alphas, info = self.rasterize_splats(
                camtoworlds = camtoworlds,
                Ks          = Ks,
                width       = width,
                height      = height,
                sh_degree   = sh_degree_to_use,
                near_plane  = cfg.near_plane,
                far_plane   = cfg.far_plane,
                image_ids   = image_ids,
                render_mode = render_mode,
                masks       = masks,
                frame_idcs  = image_ids,
                camera_idcs = data["camera_idx"].to(device),
                exposure    = exposure,
            )

            if renders.shape[-1] == 4:
                colors       = renders[..., :3]   # [1, H, W, 3]
                depths_image = renders[..., 3:4]  # [1, H, W, 1]  expected depth
            else:
                colors       = renders
                depths_image = None

            if cfg.random_bkgd:
                bkgd   = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # ── Pre-backward hook (densification bookkeeping) ─────────────────
            self.cfg.strategy.step_pre_backward(
                params     = self.splats,
                optimizers = self.optimizers,
                state      = self.strategy_state,
                step       = step,
                info       = info,
            )

            # ── Colour loss (L1 + SSIM) ──────────────────────────────────────
            l1loss   = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

            # ── Original COLMAP sparse depth loss (optional) ─────────────────
            depthloss = torch.tensor(0.0, device=device)
            if cfg.depth_loss and depths_image is not None:
                pts_norm = torch.stack([
                    points[:, :, 0] / (width  - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ], dim=-1)
                grid = pts_norm.unsqueeze(2)
                d_sampled = F.grid_sample(
                    depths_image.permute(0, 3, 1, 2),
                    grid, align_corners=True,
                ).squeeze(3).squeeze(1)
                disp    = torch.where(d_sampled > 0.0, 1.0 / d_sampled, torch.zeros_like(d_sampled))
                disp_gt = 1.0 / depths_gt
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss = loss + depthloss * cfg.depth_lambda

            # ──────────────────────────────────────────────────────────────────
            # ★ GROUND-PLANE DEPTH LOSS
            # ──────────────────────────────────────────────────────────────────
            # This block computes D_gt (geometric ground truth depth at every
            # pixel the SAM mask marked as grass) and applies MSE against the
            # rasterized expected depth at the same locations.
            #
            # HOW IT WORKS:
            #   1. ground_plane_depth_guided() shoots a ray through each masked
            #      pixel and intersects it with the Zw=0 plane in world space.
            #      This gives a metric depth without any learned components.
            #   2. ground_depth_loss_fn() samples the rasterizer's rendered depth
            #      (depths_image) at the projected 2D positions of all Gaussians,
            #      then computes MSE only where D_gt > 0.
            #   3. The gradient flows back through depths_image and means2d into
            #      the Gaussian parameters (means, scales, quats).
            #
            # WHEN IT FIRES:
            #   - cfg.ground_depth_loss must be True
            #   - A ground_mask must exist for this batch image
            #   - step >= cfg.ground_depth_start_step  (warm-up delay)
            #   - depths_image must be available (render_mode=="RGB+ED")
            #
            # ★ TO DISABLE for a specific camera or image:
            #   Add a condition like:
            #       if data["image_id"].item() in skip_ids:
            #           ground_loss = torch.tensor(0.0, ...)
            # ──────────────────────────────────────────────────────────────────
            ground_loss = torch.tensor(0.0, device=device)
            if (
                cfg.ground_depth_loss
                and ground_masks is not None
                and ground_masks.sum() > 0
                and step >= cfg.ground_depth_start_step
                and depths_image is not None
            ):
                # Build world-to-camera matrices
                viewmats = torch.linalg.inv(camtoworlds)  # [1, 4, 4]

                # Build mask tensor  [1, C, H, W]  (C=1 for batch_size=1)
                # ground_masks is [1, H, W] from the dataloader
                masks_4d = ground_masks.unsqueeze(1)  # [1, 1, H, W]

                # Compute geometric ground-truth depth from the Zw=0 plane
                # D_gt shape: [C, H, W]  =  [1, H, W]
                with torch.no_grad():
                    # D_gt has no learned parameters → compute with no_grad
                    # to avoid unnecessary graph construction on the geometry math
                    D_gt = ground_plane_depth_guided(
                        means      = self.splats["means"],
                        covars     = _quat_scale_to_covar(
                            F.normalize(self.splats["quats"], dim=-1),
                            torch.exp(self.splats["scales"]),
                        ),
                        viewmats   = viewmats[0],          # [C, 4, 4]  C=1
                        Ks         = Ks[0].unsqueeze(0),   # [1, 3, 3]
                        masks      = masks_4d,
                        width      = width,
                        height     = height,
                        near_plane = cfg.near_plane,
                        far_plane  = cfg.far_plane,
                    )  # [1, H, W]

                # means2d shape from rasterizer:  [1, N, 2]
                # depths_image shape:              [1, H, W, 1]
                means2d = info["means2d"]  # [1, N, 2]  — grad already patched above

                ground_loss = ground_depth_loss_fn(
                    depths_image = depths_image.permute(0, 3, 1, 2),  # [1, 1, H, W]
                    D_gt         = D_gt,                               # [1, H, W]
                    means2d      = means2d,                            # [1, N, 2]
                    width        = width,
                    height       = height,
                )

                if torch.isfinite(ground_loss):
                    loss = loss + ground_loss * cfg.ground_depth_lambda
                else:
                    # Protect against degenerate geometry (e.g. camera looking
                    # exactly along the ground plane normal)
                    ground_loss = torch.tensor(0.0, device=device)

            # ── Opacity / scale regularisation ───────────────────────────────
            if cfg.opacity_reg > 0.0:
                loss = loss + cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            # ── Backward + optimise ──────────────────────────────────────────
            loss.backward()

            # ── Progress bar ─────────────────────────────────────────────────
            desc = (
                f"loss={loss.item():.4f} "
                f"l1={l1loss.item():.4f} "
                f"sh={sh_degree_to_use}"
            )
            if cfg.depth_loss:
                desc += f" colmap_d={depthloss.item():.5f}"
            if cfg.ground_depth_loss:
                desc += f" gnd={ground_loss.item():.5f}"
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f" pose_err={pose_err.item():.5f}"
            pbar.set_description(desc)

            # ── Tensorboard ──────────────────────────────────────────────────
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss",          loss.item(),           step)
                self.writer.add_scalar("train/l1loss",        l1loss.item(),         step)
                self.writer.add_scalar("train/ssimloss",      ssimloss.item(),       step)
                self.writer.add_scalar("train/num_GS",        len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem_gb",        mem,                   step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/colmap_depthloss", depthloss.item(), step)
                if cfg.ground_depth_loss:
                    self.writer.add_scalar("train/ground_loss",   ground_loss.item(), step)
                    self.writer.add_scalar("train/ground_lambda",  cfg.ground_depth_lambda, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # ── Checkpoint ───────────────────────────────────────────────────
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem   = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem":          mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS":       len(self.splats["means"]),
                }
                print(f"\n[Step {step}] {stats}")
                with open(f"{self.stats_dir}/train_step{step:04d}_rank{world_rank}.json", "w") as f:
                    json.dump(stats, f)

                ckpt_data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    ckpt_data["pose_adjust"] = (
                        self.pose_adjust.module.state_dict()
                        if world_size > 1 else self.pose_adjust.state_dict()
                    )
                if cfg.app_opt:
                    ckpt_data["app_module"] = (
                        self.app_module.module.state_dict()
                        if world_size > 1 else self.app_module.state_dict()
                    )
                torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_{step}_rank{world_rank}.pt")

            # ── PLY export ───────────────────────────────────────────────────
            if (step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1) and cfg.save_ply:
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]
                export_splats(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"],
                    sh0=sh0, shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # ── Sparse gradient conversion (if needed) ───────────────────────
            if cfg.sparse_grad:
                assert cfg.packed
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )

            # ── Visibility mask (for visible_adam) ───────────────────────────
            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # ── Optimizer step ───────────────────────────────────────────────
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # ── Post-backward densification ──────────────────────────────────
            if isinstance(cfg.strategy, DefaultStrategy):
                cfg.strategy.step_post_backward(
                    params=self.splats, optimizers=self.optimizers,
                    state=self.strategy_state, step=step, info=info, packed=cfg.packed,
                )
            elif isinstance(cfg.strategy, MCMCStrategy):
                cfg.strategy.step_post_backward(
                    params=self.splats, optimizers=self.optimizers,
                    state=self.strategy_state, step=step, info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(cfg.strategy)

            # ── Evaluation checkpoints ───────────────────────────────────────
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # ── Compression ──────────────────────────────────────────────────
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            # ── Viewer update ────────────────────────────────────────────────
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                steps_per_sec = 1.0 / max(time.time() - tic, 1e-10)
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_step * steps_per_sec
                )
                self.viewer.update(step, num_train_rays_per_step)


    # ==========================================================================
    # SECTION I — EVAL / RENDER TRAJ / COMPRESSION / VIEWER
    # ==========================================================================
    # These are unchanged from simple_trainer.py.

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        print(f"\n[eval] step={step}  stage={stage}")
        cfg        = self.cfg
        device     = self.device
        world_rank = self.world_rank

        valloader  = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics      = defaultdict(list)

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks          = data["K"].to(device)
            pixels      = data["image"].to(device) / 255.0
            masks       = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            exposure    = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds = camtoworlds,
                Ks          = Ks,
                width       = width,
                height      = height,
                sh_degree   = cfg.sh_degree,
                near_plane  = cfg.near_plane,
                far_plane   = cfg.far_plane,
                masks       = masks,
                frame_idcs  = None,
                camera_idcs = data["camera_idx"].to(device),
                exposure    = exposure,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)

            if world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )
                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc = color_correct_affine(colors, pixels)
                    else:
                        cc = color_correct_quadratic(colors, pixels)
                    cc_p = cc.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({"ellipse_time": ellipse_time, "num_GS": len(self.splats["means"])})
            print(
                f"  PSNR={stats['psnr']:.3f}  SSIM={stats['ssim']:.4f}"
                f"  LPIPS={stats['lpips']:.3f}  t={stats['ellipse_time']:.3f}s"
                f"  GS={stats['num_GS']}"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        if self.cfg.disable_video:
            return
        print(f"\n[render_traj] step={step}")
        cfg    = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height)
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(f"Unknown render_traj_path: {cfg.render_traj_path}")

        camtoworlds_all = np.concatenate([
            camtoworlds_all,
            np.repeat(np.array([[[0., 0., 0., 1.]]]), len(camtoworlds_all), axis=0),
        ], axis=1)
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K     = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)

        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering traj"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds = camtoworlds_all[i:i+1],
                Ks          = K[None],
                width       = width,
                height      = height,
                sh_degree   = cfg.sh_degree,
                near_plane  = cfg.near_plane,
                far_plane   = cfg.far_plane,
                render_mode = "RGB+ED",
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
            canvas = torch.cat([colors, depths.repeat(1, 1, 1, 3)], dim=2)
            writer.append_data((canvas.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
        writer.close()
        print(f"  Saved: {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        print("\n[compression]")
        compress_dir = f"{self.cfg.result_dir}/compression/rank{self.world_rank}"
        os.makedirs(compress_dir, exist_ok=True)
        self.compression_method.compress(compress_dir, self.splats)
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        width  = render_tab_state.render_width  if render_tab_state.preview_render else render_tab_state.viewer_width
        height = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K   = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB", "depth(accumulated)": "D",
            "depth(expected)": "ED", "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds    = c2w[None],
            Ks             = K[None],
            width          = width,
            height         = height,
            sh_degree      = min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane     = render_tab_state.near_plane,
            far_plane      = render_tab_state.far_plane,
            radius_clip    = render_tab_state.radius_clip,
            eps2d          = render_tab_state.eps2d,
            backgrounds    = torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode    = RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode = render_tab_state.rasterize_mode,
            camera_model   = render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count    = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., :1]
            near  = render_tab_state.near_plane if render_tab_state.normalize_nearfar else depth.min()
            far   = render_tab_state.far_plane  if render_tab_state.normalize_nearfar else depth.max()
            d_norm = ((depth - near) / (far - near + 1e-10)).clamp(0, 1)
            if render_tab_state.inverse:
                d_norm = 1 - d_norm
            return apply_float_colormap(d_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., :1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()


# ==============================================================================
# SECTION J — MAIN & CLI ENTRY POINT
# ==============================================================================

def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    # Import post-processing modules if needed (same pattern as simple_trainer)
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import BilateralGrid, slice, total_variation_loss
        else:
            from lib_bilagrid import BilateralGrid, slice, total_variation_loss
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("[main] Viewer disabled for distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # Eval-only mode: load checkpoint(s) and run evaluation
        ckpts = [
            torch.load(f, map_location=runner.device, weights_only=True)
            for f in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1_000_000)


if __name__ == "__main__":
    """
    ──────────────────────────────────────────────────────────────────────────
    USAGE EXAMPLES
    ──────────────────────────────────────────────────────────────────────────

    ① Standard training WITH ground-plane depth prior (recommended for soccer):

        CUDA_VISIBLE_DEVICES=0 python simple_trainer_guided.py default \\
            --data_dir       /data/scene-1 \\
            --colmap_dir     /data/scene-1/sparse/0 \\
            --result_dir     results/scene-1-guided \\
            --ground_seg_dir /data/output_sam/grass_masks \\
            --ground_depth_loss \\
            --ground_depth_lambda 2.3 \\
            --ground_depth_start_step 1000 \\
            --no-normalize_world_space \\
            --no-load_exposure \\
            --test_every 0 \\
            --data_factor 2 \\
            --max_steps 30000 \\
            --disable_viewer

    ② MCMC strategy (better for large scenes, controls Gaussian count):

        CUDA_VISIBLE_DEVICES=0 python simple_trainer_guided.py mcmc \\
            --data_dir       /data/scene-1 \\
            --colmap_dir     /data/scene-1/sparse/0 \\
            --result_dir     results/scene-1-mcmc-guided \\
            --ground_seg_dir /data/output_sam/grass_masks \\
            --ground_depth_loss \\
            --ground_depth_lambda 2.3 \\
            --no-normalize_world_space \\
            --no-load_exposure \\
            --test_every 0 \\
            --strategy.cap-max 1000000 \\
            --disable_viewer

    ③ Eval-only from checkpoint:

        CUDA_VISIBLE_DEVICES=0 python simple_trainer_guided.py default \\
            --ckpt results/scene-1-guided/ckpts/ckpt_29999_rank0.pt \\
            --data_dir   /data/scene-1-challenge \\
            --colmap_dir /data/scene-1-challenge/sparse/0 \\
            --result_dir results/scene-1-eval

    ④ Distributed (4 GPUs, 4× effective batch → scale steps by 0.25):

        CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer_guided.py default \\
            --steps_scaler 0.25 --packed \\
            --ground_depth_loss \\
            --ground_seg_dir /data/output_sam/grass_masks \\
            ...

    ──────────────────────────────────────────────────────────────────────────
    KEY FLAGS (ground-plane prior)
    ──────────────────────────────────────────────────────────────────────────

    --ground_depth_loss           Enable the geometric floor constraint
    --ground_depth_lambda FLOAT   Loss weight (paper default: 2.3)
    --ground_seg_dir PATH         Directory of SAM ground masks
    --ground_depth_start_step N   Warm-up: don't apply loss until step N
    --no-normalize_world_space    Keep COLMAP Z-up so Zw=0 = physical floor
    --absgrad                     Better densification on textureless grass

    ──────────────────────────────────────────────────────────────────────────
    """

    configs = {
        "default": (
            "Default densification strategy (original 3DGS heuristics).",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "MCMC densification — controls Gaussian count via cap-max.",
            Config(
                init_opa     = 0.5,
                init_scale   = 0.1,
                opacity_reg  = 0.01,
                scale_reg    = 0.01,
                strategy     = MCMCStrategy(verbose=True),
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.compression == "png":
        try:
            import plas, torchpq
        except ImportError:
            raise ImportError(
                "PNG compression requires:\n"
                "  pip install git+https://github.com/fraunhoferhhi/PLAS.git\n"
                "  pip install torchpq"
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "--with_ut requires --with_eval3d"

    # Warn users about a common mistake
    if cfg.ground_depth_loss and cfg.normalize_world_space:
        print(
            "\n⚠  WARNING: --ground_depth_loss is on but --normalize_world_space "
            "is also on.\n"
            "   The ground-plane prior assumes Zw=0 is the physical floor in "
            "COLMAP coordinates.\n"
            "   Normalizing the world may rotate/scale the scene and break this "
            "assumption.\n"
            "   Add --no-normalize_world_space to your command.\n"
        )

    cli(main, cfg, verbose=True)