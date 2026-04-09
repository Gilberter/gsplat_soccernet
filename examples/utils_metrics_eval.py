# FILE: eval_challenge_fixed.py
"""
Evaluation script for SoccerNet challenge with multi-checkpoint support,
metrics tracking, and training history analysis.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import cv2
import numpy as np
import torch
import imageio.v3 as iio
from tqdm import tqdm

from utils_eval_challenge import (
    load_challenges,
    save_outputs,
    render_camera,
    load_splats,
    save_outputs_canvas,
)

# ============================================================================
# DATA STRUCTURES FOR METRICS & HISTORY
# ============================================================================

@dataclass
class EvalMetrics:
    """Metrics for a single evaluation run"""
    step: int
    checkpoint: str
    psnr: float
    ssim: float
    lpips: Optional[float] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "checkpoint": self.checkpoint,
            "psnr": round(self.psnr, 4),
            "ssim": round(self.ssim, 4),
            "lpips": round(self.lpips, 4) if self.lpips else None,
            "timestamp": self.timestamp,
        }


class MetricsTracker:
    """Track evaluation metrics across multiple checkpoints"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_list: List[EvalMetrics] = []
        self.history_file = os.path.join(output_dir, "eval_history.json")
        self.metrics_csv = os.path.join(output_dir, "eval_metrics.csv")
        
        # Load existing history if it exists
        self._load_history()
    
    def _load_history(self):
        """Load existing evaluation history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.metrics_list = data.get("metrics", [])
                print(f"✓ Loaded existing history with {len(self.metrics_list)} entries")
            except Exception as e:
                print(f"⚠️  Could not load history: {e}")
    
    def add_metrics(self, metrics: EvalMetrics):
        """Add metrics to tracker"""
        self.metrics_list.append(metrics)
    
    def save(self):
        """Save metrics to JSON and CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save as JSON
        data = {
            "total_evals": len(self.metrics_list),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": [m.to_dict() for m in self.metrics_list],
        }
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved metrics history to {self.history_file}")
        
        # Save as CSV for easy viewing
        self._save_csv()
    
    def _save_csv(self):
        """Save metrics as CSV for easy viewing in spreadsheet"""
        with open(self.metrics_csv, "w") as f:
            # Header
            f.write("Step,Checkpoint,PSNR,SSIM,LPIPS,Timestamp\n")
            # Rows
            for m in self.metrics_list:
                lpips_str = f"{m.lpips:.4f}" if m.lpips else "N/A"
                f.write(f"{m.step},{os.path.basename(m.checkpoint)},{m.psnr:.4f},{m.ssim:.4f},{lpips_str},{m.timestamp}\n")
        print(f"✓ Saved metrics CSV to {self.metrics_csv}")
    
    def print_summary(self):
        """Print a summary of all evaluations"""
        if not self.metrics_list:
            print("No evaluation history")
            return
        
        print("\n" + "="*80)
        print("📊 EVALUATION HISTORY SUMMARY")
        print("="*80)
        print(f"{'Step':<8} {'Checkpoint':<40} {'PSNR':<10} {'SSIM':<10}")
        print("-"*80)
        
        for m in sorted(self.metrics_list, key=lambda x: x.step):
            ckpt_name = os.path.basename(m.checkpoint)
            print(f"{m.step:<8} {ckpt_name:<40} {m.psnr:<10.4f} {m.ssim:<10.4f}")
        
        # Best metrics
        best_psnr = max(self.metrics_list, key=lambda x: x.psnr)
        best_ssim = max(self.metrics_list, key=lambda x: x.ssim)
        
        print("="*80)
        print(f"Best PSNR: {best_psnr.psnr:.4f} at step {best_psnr.step}")
        print(f"Best SSIM: {best_ssim.ssim:.4f} at step {best_ssim.step}")
        print("="*80 + "\n")


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

class MetricsComputer:
    """Compute PSNR, SSIM, LPIPS for evaluation"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metric computation functions"""
        try:
            from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            
            self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self.lpips_fn = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
            self.has_lpips = True
        except ImportError:
            print("⚠️  torchmetrics not available. Computing PSNR/SSIM only.")
            self.has_lpips = False
    
    @torch.no_grad()
    def compute(
        self,
        render: torch.Tensor,  # [H, W, 3] float [0, 1]
        gt: torch.Tensor,      # [H, W, 3] float [0, 1]
    ) -> Tuple[float, float, Optional[float]]:
        """Compute PSNR, SSIM, LPIPS"""
        
        # Convert to [1, 3, H, W]
        render_p = render.permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt_p = gt.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        psnr = self.psnr_fn(render_p, gt_p).item()
        ssim = self.ssim_fn(render_p, gt_p).item()
        lpips = None
        
        if self.has_lpips:
            try:
                lpips = self.lpips_fn(render_p, gt_p).item()
            except Exception as e:
                print(f"⚠️  LPIPS computation failed: {e}")
        
        return psnr, ssim, lpips


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_checkpoint(
    ckpt_path: str,
    data_dir: str,
    output_dir: str,
    device: str,
    max_sh_degree: int = 3,
    indices: Optional[List[int]] = None,
    near: float = 0.1,
    far: float = 1000.0,
    compute_metrics: bool = True,
    save_renders: bool = True,
) -> Dict:
    """
    Evaluate a single checkpoint on challenge views.
    
    Returns:
        Dictionary with metrics and metadata
    """
    
    print(f"\n{'='*80}")
    print(f"🔍 EVALUATING CHECKPOINT: {os.path.basename(ckpt_path)}")
    print(f"{'='*80}\n")
    
    # Load Gaussians from checkpoint
    means, quats, scales, opacities, colors, sh_degree, app_module, features = load_splats(
        [ckpt_path], device
    )
    
    print(f"✓ Loaded Gaussians:")
    print(f"  Count: {len(means):,}")
    print(f"  Opacity range: [{opacities.min():.4f}, {opacities.max():.4f}]")
    print(f"  Colors range: [{colors.min():.4f}, {colors.max():.4f}]")
    
    # Determine SH degree to use
    if sh_degree is not None:
        sh_degree_render = min(max_sh_degree, sh_degree)
    else:
        sh_degree_render = None
    
    # Load challenge cameras
    c2w_mats, ks_list, imsize_list, image_ids = load_challenges(
        f"{data_dir}/sparse/0", factor=2
    )
    n_cameras = len(image_ids)
    
    # Determine which cameras to evaluate
    if indices is not None and len(indices) == 2:
        start, end = indices
        eval_indices = np.arange(start, min(end + 1, n_cameras))
    elif indices is not None:
        eval_indices = np.array(indices)
        eval_indices = eval_indices[eval_indices < n_cameras]
    else:
        eval_indices = np.arange(n_cameras)
    
    print(f"✓ Loaded {n_cameras} camera(s), evaluating {len(eval_indices)}\n")
    
    # Setup output directories
    os.makedirs(f"{output_dir}/renders", exist_ok=True)
    os.makedirs(f"{output_dir}/comparisons", exist_ok=True)
    
    # Load ground truth renders if available
    gt_dir = os.path.join(data_dir, "renders")
    has_gt = os.path.exists(gt_dir)
    if has_gt:
        gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])
    
    # Initialize metrics computer
    if compute_metrics and has_gt:
        metrics_computer = MetricsComputer(device=device)
    
    # Evaluation loop
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    print(f"🎨 Rendering and evaluating {len(eval_indices)} views...")
    pbar = tqdm(enumerate(eval_indices), total=len(eval_indices))
    
    for i, idx in pbar:
        c2w = c2w_mats[idx]
        K = ks_list[idx].copy()
        width, height = imsize_list[idx]
        
        # Render
        render_out, n_gaussians = render_camera(
            means, quats, scales, opacities, colors,
            sh_degree_render, c2w, K,
            width, height,
            near=near, far=far,
            device=device,
            app_module=app_module,
            features=features,
        )
        
        # Post-process render output
        if isinstance(render_out, torch.Tensor):
            render_out = render_out.detach().cpu().numpy()
        
        if render_out.ndim == 4:
            render_out = render_out.squeeze(0)
        
        if render_out.shape[-1] > 3:
            render_out = render_out[..., :3]
        
        if render_out.dtype != np.uint8:
            render_out = np.clip(render_out * 255.0, 0, 255).astype(np.uint8)
        
        tag = f"{i:05d}"
        
        # Save render
        if save_renders:
            save_outputs(output_dir, tag, render_out)
        
        # Compute metrics
        psnr, ssim, lpips = None, None, None
        if compute_metrics and has_gt and i < len(gt_paths):
            gt_img = iio.imread(gt_paths[i])
            gt_img_torch = torch.from_numpy(gt_img).float().to(device) / 255.0
            render_torch = torch.from_numpy(render_out).float().to(device) / 255.0
            
            psnr, ssim, lpips = metrics_computer.compute(render_torch, gt_img_torch)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            if lpips is not None:
                all_lpips.append(lpips)
            
            # Save comparison
            save_outputs_canvas(output_dir, render_out, gt_img, tag)
            
            pbar.set_description(
                f"[{i+1}/{len(eval_indices)}] PSNR: {psnr:.2f} | SSIM: {ssim:.3f}"
            )
        else:
            pbar.set_description(f"[{i+1}/{len(eval_indices)}] Rendered")
    
    # Compute averages
    results = {
        "checkpoint": ckpt_path,
        "num_gaussians": len(means),
        "num_views_evaluated": len(eval_indices),
    }
    
    if all_psnr:
        results["psnr_mean"] = float(np.mean(all_psnr))
        results["psnr_std"] = float(np.std(all_psnr))
        results["ssim_mean"] = float(np.mean(all_ssim))
        results["ssim_std"] = float(np.std(all_ssim))
        
        if all_lpips:
            results["lpips_mean"] = float(np.mean(all_lpips))
            results["lpips_std"] = float(np.std(all_lpips))
        
        print(f"\n✓ Results:")
        print(f"  PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f}")
        print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        if all_lpips:
            print(f"  LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    
    return results


# ============================================================================
# MULTI-CHECKPOINT EVALUATION
# ============================================================================

def main(args):
    """Main evaluation function supporting multiple checkpoints"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Setup output directory
    if args.specific:
        output_dir = args.result_folder
    else:
        output_dir = os.path.join(args.data_dir, args.result_folder)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(output_dir)
    
    # Process checkpoints
    print(f"\n{'='*80}")
    print(f"📋 CHECKPOINT SUMMARY")
    print(f"{'='*80}")
    print(f"Total checkpoints to evaluate: {len(args.ckpt)}")
    for i, ckpt in enumerate(args.ckpt, 1):
        print(f"  {i}. {os.path.basename(ckpt)}")
    print()
    
    # Evaluate each checkpoint
    for ckpt_path in args.ckpt:
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            continue
        
        # Create subdirectory for this checkpoint
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        ckpt_output_dir = os.path.join(output_dir, ckpt_name)
        os.makedirs(ckpt_output_dir, exist_ok=True)
        
        try:
            # Evaluate checkpoint
            results = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                data_dir=args.data_dir,
                output_dir=ckpt_output_dir,
                device=device,
                max_sh_degree=args.max_sh_degree,
                indices=args.indices,
                near=args.near,
                far=args.far,
                compute_metrics=not args.no_metrics,
                save_renders=not args.no_save,
            )
            
            # Extract step from checkpoint name (e.g., ckpt_39999_rank0.pt -> 39999)
            try:
                step = int(ckpt_name.split("_")[1])
            except:
                step = 0
            
            # Track metrics
            if "psnr_mean" in results:
                metrics = EvalMetrics(
                    step=step,
                    checkpoint=ckpt_path,
                    psnr=results["psnr_mean"],
                    ssim=results["ssim_mean"],
                    lpips=results.get("lpips_mean"),
                )
                metrics_tracker.add_metrics(metrics)
            
            # Copy checkpoint to output
            import shutil
            dst = os.path.join(ckpt_output_dir, os.path.basename(ckpt_path))
            if not os.path.exists(dst):
                shutil.copy2(ckpt_path, dst)
                print(f"✓ Checkpoint copied to {ckpt_name}/")
        
        except Exception as e:
            print(f"❌ Error evaluating {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save and print summary
    print(f"\n{'='*80}")
    print(f"💾 SAVING EVALUATION HISTORY")
    print(f"{'='*80}\n")
    metrics_tracker.save()
    metrics_tracker.print_summary()
    
    print(f"✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SoccerNet challenge with multi-checkpoint support"
    )
    
    # Checkpoints
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to checkpoint .pt files (supports multiple)",
    )
    
    # Scene
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to challenge directory (contains sparse/ and renders/)",
    )
    
    # Output
    parser.add_argument(
        "--result_folder",
        type=str,
        default="eval_results",
        help="Output folder name (relative to data_dir or absolute if --specific)",
    )
    parser.add_argument(
        "--specific",
        action="store_true",
        help="Treat --result_folder as absolute path",
    )
    
    # Rendering options
    parser.add_argument(
        "--max_sh_degree",
        type=int,
        default=3,
        help="Maximum SH degree to use",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.1,
        help="Near plane distance",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=1000.0,
        help="Far plane distance",
    )
    
    # Camera selection
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Camera indices (2 values = range, >2 = explicit list, none = all)",
    )
    
    # Evaluation options
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip metric computation (faster)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving render images",
    )
    
    args = parser.parse_args()
    main(args)