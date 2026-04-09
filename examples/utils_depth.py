"""
depth_supervision.py
====================
Combined depth supervision using:
  1. COLMAP sparse points  — correct metric scale, used to align DA3
  2. DA3 dense depth map   — wrong scale but dense, aligned via COLMAP anchor

Strategy
--------
  Step A: compute per-image scale s and shift t so that
          s * DA3[u,v] + t ≈ COLMAP_depth   at sparse COLMAP pixels.
          This brings DA3 into COLMAP metric space.

  Step B: build a dense pseudo-GT depth map:
          D_pseudo[u,v] = s * DA3[u,v] + t   everywhere

  Step C: compute losses:
          - Sparse L1  : rendered depth vs COLMAP at sparse pixels (high trust)
          - Dense  L1  : rendered depth vs D_pseudo everywhere     (lower trust)

Both losses operate in the same metric space so gradients are coherent.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict

import numpy as np
import torch.nn as nn
# =============================================================================
# SCALE ALIGNMENT — COLMAP anchors DA3
# =============================================================================

def align_da3_to_colmap(
    da3_depth:      Tensor,          # (H, W)   DA3 raw depth (wrong scale)
    points_px:      Tensor,          # (M, 2)   COLMAP pixel coords (x, y)
    depths_colmap:  Tensor,          # (M,)     COLMAP metric depths
    min_points:     int   = 5,
    clamp_scale:    Tuple[float, float] = (0.05, 20.0),
) -> Tuple[float, float, bool]:
    """
    Find s, t such that  s * DA3[u,v] + t ≈ COLMAP_depth
    at the M sparse COLMAP pixel locations.

    Returns
    -------
    s       : scale factor   (float)
    t       : shift          (float)
    success : whether we had enough valid points to fit
    """
    if points_px.shape[0] < min_points:
        return 1.0, 0.0, False

    H, W = da3_depth.shape
    xs = points_px[:, 0].long().clamp(0, W - 1)
    ys = points_px[:, 1].long().clamp(0, H - 1)

    da3_at_pts = da3_depth[ys, xs]                    # (M,)  DA3 values at COLMAP pixels
    col_depths  = depths_colmap                        # (M,)  COLMAP metric depths

    valid = (da3_at_pts > 0) & (col_depths > 0)
    if valid.sum() < min_points:
        return 1.0, 0.0, False

    x = da3_at_pts[valid].float()                     # DA3 values  (predictor)
    y = col_depths[valid].float()                      # COLMAP depths (target)

    # Closed-form weighted least squares: min ||s*x + t - y||²
    x_mean = x.mean()
    y_mean = y.mean()
    x_var  = ((x - x_mean) ** 2).mean().clamp(min=1e-8)
    s      = ((x - x_mean) * (y - y_mean)).mean() / x_var
    t      = y_mean - s * x_mean

    # Safety clamp — if fit is degenerate, fall back to identity
    s = float(s.item())
    t = float(t.item())
    s = float(torch.tensor(s).clamp(*clamp_scale).item())

    return s, t, True


def build_aligned_depth(
    da3_depth: Tensor,    # (H, W)
    s: float,
    t: float,
) -> Tensor:
    """Apply scale+shift to produce metric-aligned dense depth map."""
    return (s * da3_depth + t).clamp(min=0.0)        # (H, W)


# =============================================================================
# COMBINED DEPTH LOSS
# =============================================================================

def combined_depth_loss(
    depth_rendered: Tensor,          # (1, H, W, 1)  from rasterization RGB+ED
    da3_depth:      Tensor,          # (1, 1, H, W)  raw DA3 from dataloader
    points_px:      Tensor,          # (1, M, 2)     COLMAP pixel coords
    depths_colmap:  Tensor,          # (1, M)        COLMAP metric depths
    scene_scale:    float,
    lambda_sparse:  float = 1.0,     # weight for COLMAP sparse loss
    lambda_dense:   float = 0.1,     # weight for DA3 dense loss (lower trust)
    min_colmap_pts: int   = 5,
    device:         str   = "cuda",
) -> Tuple[Tensor, Dict]:
    """
    Compute combined sparse (COLMAP) + dense (DA3-aligned) depth loss.

    Parameters
    ----------
    depth_rendered  : (1, H, W, 1) rendered expected depth, requires_grad=True
    da3_depth       : (1, 1, H, W) raw DA3 depth from dataloader
    points_px       : (1, M, 2)    COLMAP projected pixel positions
    depths_colmap   : (1, M)       COLMAP metric depths at those positions
    scene_scale     : parser.scene_scale * 1.1 * global_scale
    lambda_sparse   : loss weight for sparse COLMAP supervision
    lambda_dense    : loss weight for dense DA3 supervision
    min_colmap_pts  : minimum COLMAP points needed to fit alignment

    Returns
    -------
    loss     : scalar tensor (backprop-able)
    info     : dict with individual loss components for logging
    """
    B, H, W, _ = depth_rendered.shape
    assert B == 1, "batch_size must be 1"

    dr = depth_rendered[0, :, :, 0]               # (H, W)  live tensor, grad flows
    da3 = da3_depth[0, 0].float().to(device)       # (H, W)  no grad needed
    pts = points_px[0].to(device)                  # (M, 2)
    dc  = depths_colmap[0].float().to(device)      # (M,)

    loss_sparse = torch.tensor(0.0, device=device)
    loss_dense  = torch.tensor(0.0, device=device)
    info = {
        "loss_sparse": 0.0,
        "loss_dense":  0.0,
        "n_sparse":    0,
        "n_dense":     0,
        "da3_scale_s": 1.0,
        "da3_shift_t": 0.0,
        "aligned":     False,
    }

    # ------------------------------------------------------------------
    # STEP A: align DA3 to COLMAP metric scale
    # ------------------------------------------------------------------
    s, t, aligned = align_da3_to_colmap(
        da3_depth     = da3,
        points_px     = pts,
        depths_colmap = dc,
        min_points    = min_colmap_pts,
    )
    info["da3_scale_s"] = s
    info["da3_shift_t"] = t
    info["aligned"]     = aligned

    if not aligned:
        # Not enough COLMAP points — fall back to sparse-only loss
        # (no DA3 this step)
        pass

    da3_metric = build_aligned_depth(da3, s, t)    # (H, W)  metric scale

    # ------------------------------------------------------------------
    # STEP B: Sparse COLMAP loss
    # Supervise rendered depth at the exact COLMAP pixel positions.
    # These are ground-truth metric depths — highest trust.
    # ------------------------------------------------------------------
    xs = pts[:, 0].long().clamp(0, W - 1)
    ys = pts[:, 1].long().clamp(0, H - 1)

    dr_at_pts  = dr[ys, xs]                         # (M,)  rendered at COLMAP pixels
    valid_col  = dc > 0
    n_sparse   = int(valid_col.sum().item())

    if n_sparse >= 2:
        loss_sparse = (
            F.l1_loss(dr_at_pts[valid_col], dc[valid_col])
            * scene_scale
        )
        info["loss_sparse"] = loss_sparse.item()
        info["n_sparse"]    = n_sparse

    # ------------------------------------------------------------------
    # STEP C: Dense DA3 loss (aligned to metric scale)
    # Supervise all pixels where rendered depth is valid.
    # Lower weight because DA3 alignment is approximate.
    # ------------------------------------------------------------------
    if aligned:
        valid_dense = (dr > 0) & (da3_metric > 0)
        n_dense = int(valid_dense.sum().item())

        if n_dense > 100:
            loss_dense = (
                F.l1_loss(dr[valid_dense], da3_metric[valid_dense])
                * scene_scale
            )
            info["loss_dense"] = loss_dense.item()
            info["n_dense"]    = n_dense

    # ------------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------------
    total = lambda_sparse * loss_sparse + lambda_dense * loss_dense

    return total, info


# =============================================================================
# CONVENIENCE WRAPPER — drop-in for the trainer
# =============================================================================

def depth_loss_step(
    depth_rendered: Tensor,          # (1, H, W, 1)
    da3_depth:      Tensor,          # (1, 1, H, W)  raw from dataloader
    points_px:      Tensor,          # (1, M, 2)
    depths_colmap:  Tensor,          # (1, M)
    scene_scale:    float,
    lambda_sparse:  float = 1.0,
    lambda_dense:   float = 0.1,
    device:         str   = "cuda",
) -> Tuple[Tensor, Dict]:
    """
    Single-call interface for use inside the training loop.
    Handles all edge cases (missing data, empty tensors).

    Usage in trainer:
        if cfg.depth_loss:
            depthloss, depth_info = depth_loss_step(
                depth_rendered = depth_rendered,       # (1, H, W, 1)
                da3_depth      = depth_prior,          # (1, 1, H, W)  before permute
                points_px      = points,               # (1, M, 2)
                depths_colmap  = depths_gt,            # (1, M)
                scene_scale    = self.scene_scale,
                lambda_sparse  = cfg.depth_lambda,
                lambda_dense   = cfg.depth_lambda * 0.1,
                device         = device,
            )
            if torch.isfinite(depthloss) and depthloss > 0:
                loss = loss + depthloss
    """
    # Guard: need rendered depth
    if depth_rendered is None:
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return zero, {"loss_sparse": 0., "loss_dense": 0., "aligned": False}

    # Guard: empty COLMAP points
    if points_px.shape[1] == 0 or depths_colmap.shape[1] == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return zero, {"loss_sparse": 0., "loss_dense": 0., "aligned": False,
                      "n_sparse": 0, "n_dense": 0}

    try:
        return combined_depth_loss(
            depth_rendered = depth_rendered,
            da3_depth      = da3_depth,
            points_px      = points_px,
            depths_colmap  = depths_colmap,
            scene_scale    = scene_scale,
            lambda_sparse  = lambda_sparse,
            lambda_dense   = lambda_dense,
            device         = device,
        )
    except Exception as e:
        print(f"[depth_loss_step] Error: {e}")
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return zero, {"loss_sparse": 0., "loss_dense": 0., "aligned": False, "error": str(e)}


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def reduction_batch_based(image_loss, M):

    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def reduction_image_based(image_loss, M):

    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)
class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.reduction)

class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

        self.scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.reduction)

        return total



class GradientLossMemoryEfficient(nn.Module):
    """
    Simplified gradient loss that works at single scale.
    This avoids the 4x memory explosion from multi-scale processing.
    """
    def __init__(self, reduction='batch-based'):
        super().__init__()
        if reduction == 'batch-based':
            self.reduction = self._reduction_batch_based
        else:
            self.reduction = self._reduction_image_based

    @staticmethod
    def _reduction_batch_based(image_loss, M):
        divisor = torch.sum(M)
        if divisor == 0:
            return torch.tensor(0.0, device=image_loss.device)
        return torch.sum(image_loss) / divisor

    @staticmethod
    def _reduction_image_based(image_loss, M):
        valid = M.nonzero()
        if len(valid) > 0:
            image_loss[valid] = image_loss[valid] / M[valid]
        return torch.mean(image_loss)

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            prediction: (B, H, W) predicted depth
            target:     (B, H, W) target depth
            mask:       (B, H, W) validity mask (binary)
        
        Returns:
            Scalar loss
        """
        assert prediction.ndim == 3, f"Expected (B, H, W), got {prediction.shape}"
        assert prediction.shape == target.shape == mask.shape, \
            f"Shape mismatch! Pred: {prediction.shape}, Target: {target.shape}, Mask: {mask.shape}"
        assert all(s > 0 for s in prediction.shape), f"Zero-dimension detected in shape: {prediction.shape}"

        M = torch.sum(mask, dim=(1, 2), keepdim=False)  # (B,)
        
        # Compute differences
        diff = prediction - target
        diff = torch.mul(mask, diff)

        # Gradients in X direction
        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        # Gradients in Y direction
        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, dim=(1, 2)) + torch.sum(grad_y, dim=(1, 2))

        return self.reduction(image_loss, M)


class ScaleAndShiftInvariantLossLight(nn.Module):
    """
    SSI Loss with LIGHT gradient regularization.
    - Single scale only (no multi-scale loop)
    - Smaller stride to reduce computation
    - Lower alpha weight
    """
    
    def __init__(self, alpha=0.5, reduction='batch-based'):
        super().__init__()
        self.data_loss = MSELoss(reduction=reduction)
        self.gradient_loss = GradientLossMemoryEfficient(reduction=reduction)
        self.alpha = alpha
        self.prediction_ssi = None

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute SSI loss with light gradient regularization."""
        assert prediction.ndim == 3, f"Expected (B, H, W), got {prediction.shape}"
        assert prediction.shape == target.shape == mask.shape, \
            f"Shape mismatch! Pred: {prediction.shape}, Target: {target.shape}, Mask: {mask.shape}"
        # Compute scale and shift
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        
        # Apply alignment
        self.prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        
        # Data loss (primary)
        loss = self.data_loss(self.prediction_ssi, target, mask)
        
        # Light gradient loss (downsampled to reduce memory)
        if self.alpha > 0:
            # Downsample by factor of 2 to reduce memory
            step = 2
            pred_ds = self.prediction_ssi[:, ::step, ::step]
            target_ds = target[:, ::step, ::step]
            mask_ds = mask[:, ::step, ::step]
            
            grad_loss = self.gradient_loss(pred_ds, target_ds, mask_ds)
            loss = loss + self.alpha * grad_loss
        
        return loss

    def get_prediction_ssi(self):
        return self.prediction_ssi



class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=3.0, scales=1, reduction='batch-based'):
        super().__init__()

        self.data_loss = MSELoss(reduction=reduction)
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.alpha = alpha

        self.prediction_ssi = None

    def forward(self, prediction, target, mask):
        
        scale, shift = compute_scale_and_shift(prediction, target, mask)

        self.prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.data_loss(self.prediction_ssi, target, mask)
        if self.alpha > 0:
            total += self.alpha * self.regularization_loss(self.prediction_ssi, target, mask)

        return total

    def get_prediction_ssi(self):
        return self.prediction_ssi



class ScaleAndShiftInvariantDALoss(nn.Module):
    def __init__(self, grad_matching, **kargs):
        super(ScaleAndShiftInvariantDALoss, self).__init__()
        self.grad_matching = grad_matching
        self.scaled_prediction = None

    def forward(self, prediction, target, mask, min_depth=None, max_depth=None, **kwargs):
        
        #_, h_i, w_i = prediction.shape
        #_, h_t, w_t = target.shape
    
        #if h_i != h_t or w_i != w_t:
        #    prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        #prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze().bool()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        
        self.scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        scale_target = target
        
        sampling_mask = mask
        if self.grad_matching:
            N = torch.sum(sampling_mask)
            d_diff = self.scaled_prediction - scale_target
            d_diff = torch.mul(d_diff, sampling_mask)

            v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
            v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
            h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
            loss = gradient_loss / N
        else:
            loss = nn.functional.l1_loss(self.scaled_prediction[mask], scale_target[mask])
            
        return loss

    def get_prediction_ssi(self):
        return self.scaled_prediction


def loss_dict():

    loss_fns = {
        'ssimse': ScaleAndShiftInvariantLoss(),
        'ssimse_nogradmatch': ScaleAndShiftInvariantLoss(alpha=0),
        'ssimae': TrimmedProcrustesLoss(),
        'ssigm': ScaleAndShiftInvariantDALoss(grad_matching=True),
        'silog': SiLogLoss(),
    }

    return loss_fns

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
        : int(len(res) * (1.0 - trim))
    ]

    return trimmed.sum() / (2 * M.sum())

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def compute_hdn_loss(SSI_LOSS, depth_preds, depth_gt, mask_valid_list):

    num_contexts = mask_valid_list.shape[0]
    repeated_preds = depth_preds.unsqueeze(0).repeat(num_contexts, 1, 1, 1, 1)
    repeated_gt = depth_gt.unsqueeze(0).repeat(num_contexts, 1, 1, 1, 1)
    
    repeated_preds = repeated_preds.reshape(-1, *depth_preds.shape[-3:])
    repeated_gt = repeated_gt.reshape(-1, *depth_gt.shape[-3:])
    repeated_masks = mask_valid_list.reshape(-1, *mask_valid_list.shape[-3:])
    
    hdn_loss_level = SSI_LOSS(repeated_preds, repeated_gt, repeated_masks)
    hdn_loss_level_list = hdn_loss_level.reshape(mask_valid_list.shape)
    hdn_loss_level_list = hdn_loss_level_list.sum(dim=0)
    
    mask_valid_list_times = mask_valid_list.sum(dim=0)
    
    valid_locations = (mask_valid_list_times != 0)
    hdn_loss_level_list[valid_locations] = (
        hdn_loss_level_list[valid_locations] / mask_valid_list_times[valid_locations]
    )
    
    final_loss = hdn_loss_level_list.sum() / mask_valid_list.sum()
    
    return final_loss