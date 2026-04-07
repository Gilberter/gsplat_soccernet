# # SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import json
# import math
# import os
# import time
# from dataclasses import dataclass, field
# from typing import Dict, List, Literal, Optional, Tuple
# from pathlib import Path

# import imageio
# import nerfview
# import numpy as np
# import torch
# import torch.nn.functional as F
# import tqdm
# import tyro
# import viser
# from datasets.colmap import Dataset, Parser
# from datasets.traj import generate_interpolated_path
# from torch import Tensor
# from torch.utils.tensorboard import SummaryWriter
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# from utils import (
#     AppearanceOptModule,
#     CameraOptModule,
#     apply_depth_colormap,
#     colormap,
#     knn,
#     rgb_to_sh,
#     set_random_seed,
# )
# from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
# from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
# from gsplat.strategy import DefaultStrategy
# from nerfview import CameraState, RenderTabState, apply_float_colormap


# @dataclass
# class Config:
#     # Disable viewer
#     disable_viewer: bool = False
#     # Path to the .pt file. If provide, it will skip training and render a video
#     ckpt: Optional[str] = None

#     # Path to the Mip-NeRF 360 dataset
#     data_dir: str = "data/360_v2/garden"
#     # Downsample factor for the dataset
#     data_factor: int = 4
#     # Directory to save results
#     result_dir: str = "results/garden"
#     # Every N images there is a test image
#     test_every: int = 8
#     # Random crop size for training  (experimental)
#     patch_size: Optional[int] = None
#     # A global scaler that applies to the scene size related parameters
#     global_scale: float = 1.0
#     # Normalize the world space
#     normalize_world_space: bool = True

#     # Port for the viewer server
#     port: int = 8080

#     # Batch size for training. Learning rates are scaled automatically
#     batch_size: int = 1
#     # A global factor to scale the number of training steps
#     steps_scaler: float = 1.0

#     # Number of training steps
#     max_steps: int = 30_000
#     # Steps to evaluate the model
#     eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
#     # Steps to save the model
#     save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

#     # Initialization strategy
#     init_type: str = "sfm"
#     # Initial number of GSs. Ignored if using sfm
#     init_num_pts: int = 100_000
#     # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
#     init_extent: float = 3.0
#     # Degree of spherical harmonics
#     sh_degree: int = 3
#     # Turn on another SH degree every this steps
#     sh_degree_interval: int = 1000
#     # Initial opacity of GS
#     init_opa: float = 0.1
#     # Initial scale of GS
#     init_scale: float = 1.0
#     # Weight for SSIM loss
#     ssim_lambda: float = 0.2

#     # Near plane clipping distance
#     near_plane: float = 0.2
#     # Far plane clipping distance
#     far_plane: float = 200

#     # GSs with opacity below this value will be pruned
#     prune_opa: float = 0.05
#     # GSs with image plane gradient above this value will be split/duplicated
#     grow_grad2d: float = 0.0002
#     # GSs with scale below this value will be duplicated. Above will be split
#     grow_scale3d: float = 0.01
#     # GSs with scale above this value will be pruned.
#     prune_scale3d: float = 0.1

#     # Start refining GSs after this iteration
#     refine_start_iter: int = 500
#     # Stop refining GSs after this iteration
#     refine_stop_iter: int = 15_000
#     # Reset opacities every this steps
#     reset_every: int = 3000
#     # Refine GSs every this steps
#     refine_every: int = 100

#     colmap_dir: str = ""

#     # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
#     packed: bool = False
#     # Use sparse gradients for optimization. (experimental)
#     sparse_grad: bool = False
#     # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
#     absgrad: bool = False
#     # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
#     antialiased: bool = False
#     # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
#     revised_opacity: bool = False

#     # Use random background for training to discourage transparency
#     random_bkgd: bool = False

#     # Enable camera optimization.
#     pose_opt: bool = False
#     # Learning rate for camera optimization
#     pose_opt_lr: float = 1e-5
#     # Regularization for camera optimization as weight decay
#     pose_opt_reg: float = 1e-6
#     # Add noise to camera extrinsics. This is only to test the camera pose optimization.
#     pose_noise: float = 0.0

#     # Enable appearance optimization. (experimental)
#     app_opt: bool = False
#     # Appearance embedding dimension
#     app_embed_dim: int = 16
#     # Learning rate for appearance optimization
#     app_opt_lr: float = 1e-3
#     # Regularization for appearance optimization as weight decay
#     app_opt_reg: float = 1e-6

#     # Enable depth loss. (experimental)
#     depth_loss: bool = False
#     # Weight for depth loss
#     depth_lambda: float = 1e-2

#     # Enable normal consistency loss. (Currently for 2DGS only)
#     normal_loss: bool = False
#     # Weight for normal loss
#     normal_lambda: float = 5e-2
#     # Iteration to start normal consistency regulerization
#     normal_start_iter: int = 7_000

#     # Distortion loss. (experimental)
#     dist_loss: bool = False
#     # Weight for distortion loss
#     dist_lambda: float = 1e-2
#     # Iteration to start distortion loss regulerization
#     dist_start_iter: int = 3_000

#     # Model for splatting.
#     model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

#     # Dump information to tensorboard every this steps
#     tb_every: int = 100
#     # Save training images to tensorboard
#     tb_save_image: bool = False

#     feature_dim: int =32

#     def adjust_steps(self, factor: float):
#         self.eval_steps = [int(i * factor) for i in self.eval_steps]
#         self.save_steps = [int(i * factor) for i in self.save_steps]
#         self.max_steps = int(self.max_steps * factor)
#         self.sh_degree_interval = int(self.sh_degree_interval * factor)
#         self.refine_start_iter = int(self.refine_start_iter * factor)
#         self.refine_stop_iter = int(self.refine_stop_iter * factor)
#         self.reset_every = int(self.reset_every * factor)
#         self.refine_every = int(self.refine_every * factor)


# def create_splats_with_optimizers(
#     parser: Parser,
#     init_type: str = "sfm",
#     init_num_pts: int = 100_000,
#     init_extent: float = 3.0,
#     init_opacity: float = 0.1,
#     init_scale: float = 1.0,
#     scene_scale: float = 1.0,
#     sh_degree: int = 3,
#     sparse_grad: bool = False,
#     batch_size: int = 1,
#     feature_dim: Optional[int] = None,
#     device: str = "cuda",
# ) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
#     if init_type == "sfm":
#         points = torch.from_numpy(parser.points).float()
#         rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
#     elif init_type == "random":
#         points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
#         rgbs = torch.rand((init_num_pts, 3))
#     else:
#         raise ValueError("Please specify a correct init_type: sfm or random")

#     N = points.shape[0]
#     # Initialize the GS size to be the average dist of the 3 nearest neighbors
#     dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
#     dist_avg = torch.sqrt(dist2_avg)
#     scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
#     quats = torch.rand((N, 4))  # [N, 4]
#     opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

#     params = [
#         # name, value, lr
#         ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
#         ("scales", torch.nn.Parameter(scales), 5e-3),
#         ("quats", torch.nn.Parameter(quats), 1e-3),
#         ("opacities", torch.nn.Parameter(opacities), 5e-2),
#     ]

#     if feature_dim is None:
#         # color is SH coefficients.
#         colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
#         colors[:, 0, :] = rgb_to_sh(rgbs)
#         params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
#         params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
#     else:
#         # features will be used for appearance and view-dependent shading
#         features = torch.rand(N, feature_dim)  # [N, feature_dim]
#         params.append(("features", torch.nn.Parameter(features), 2.5e-3))
#         colors = torch.logit(rgbs)  # [N, 3]
#         params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

#     splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
#     # Scale learning rate based on batch size, reference:
#     # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
#     # Note that this would not make the training exactly equivalent, see
#     # https://arxiv.org/pdf/2402.18824v1
#     optimizers = {
#         name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
#             [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
#             eps=1e-15 / math.sqrt(batch_size),
#             betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
#         )
#         for name, _, lr in params
#     }
#     return splats, optimizers


# class Runner:
#     """Engine for training and testing."""

#     def __init__(self, cfg: Config) -> None:
#         set_random_seed(42)

#         self.cfg = cfg
#         self.device = "cuda"

#         # Where to dump results.
#         os.makedirs(cfg.result_dir, exist_ok=True)

#         # Setup output directories.
#         self.ckpt_dir = f"{cfg.result_dir}/ckpts"
#         os.makedirs(self.ckpt_dir, exist_ok=True)
#         self.stats_dir = f"{cfg.result_dir}/stats"
#         os.makedirs(self.stats_dir, exist_ok=True)
#         self.render_dir = f"{cfg.result_dir}/renders"
#         os.makedirs(self.render_dir, exist_ok=True)

#         # Tensorboard
#         self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

#         # Load data: Training data should contain initial points and colors.
#         self.parser = Parser(
#             data_dir=cfg.data_dir,
#             factor=cfg.data_factor,
#             normalize=cfg.normalize_world_space,
#             test_every=cfg.test_every,
#             colmap_dir=cfg.colmap_dir
#         )
#         self.trainset = Dataset(
#             self.parser,
#             split="train",
#             patch_size=cfg.patch_size,
#             load_depths=cfg.depth_loss,
#         )
#         self.valset = Dataset(self.parser, split="val")
#         self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
#         print("Scene scale:", self.scene_scale)

#         # Model
#         feature_dim = 32 if cfg.app_opt else None
#         self.splats, self.optimizers = create_splats_with_optimizers(
#             self.parser,
#             init_type=cfg.init_type,
#             init_num_pts=cfg.init_num_pts,
#             init_extent=cfg.init_extent,
#             init_opacity=cfg.init_opa,
#             init_scale=cfg.init_scale,
#             scene_scale=self.scene_scale,
#             sh_degree=cfg.sh_degree,
#             sparse_grad=cfg.sparse_grad,
#             batch_size=cfg.batch_size,
#             feature_dim=feature_dim,
#             device=self.device,
#         )
#         print("Model initialized. Number of GS:", len(self.splats["means"]))
#         self.model_type = cfg.model_type

#         if self.model_type == "2dgs":
#             key_for_gradient = "gradient_2dgs"
#         else:
#             key_for_gradient = "means2d"

#         # Densification Strategy
#         self.strategy = DefaultStrategy(
#             verbose=True,
#             prune_opa=cfg.prune_opa,
#             grow_grad2d=cfg.grow_grad2d,
#             grow_scale3d=cfg.grow_scale3d,
#             prune_scale3d=cfg.prune_scale3d,
#             # refine_scale2d_stop_iter=4000, # splatfacto behavior
#             refine_start_iter=cfg.refine_start_iter,
#             refine_stop_iter=cfg.refine_stop_iter,
#             reset_every=cfg.reset_every,
#             refine_every=cfg.refine_every,
#             absgrad=cfg.absgrad,
#             revised_opacity=cfg.revised_opacity,
#             key_for_gradient=key_for_gradient,
#         )
#         self.strategy.check_sanity(self.splats, self.optimizers)
#         self.strategy_state = self.strategy.initialize_state()

#         self.pose_optimizers = []
#         if cfg.pose_opt:
#             self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
#             self.pose_adjust.zero_init()
#             self.pose_optimizers = [
#                 torch.optim.Adam(
#                     self.pose_adjust.parameters(),
#                     lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
#                     weight_decay=cfg.pose_opt_reg,
#                 )
#             ]

#         if cfg.pose_noise > 0.0:
#             self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
#             self.pose_perturb.random_init(cfg.pose_noise)

#         self.app_optimizers = []
#         if cfg.app_opt:
#             self.app_module = AppearanceOptModule(
#                 len(self.trainset), cfg.feature_dim, cfg.app_embed_dim, cfg.sh_degree
#             ).to(self.device)
#             # initialize the last layer to be zero so that the initial output is zero.
#             torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
#             torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
#             self.app_optimizers = [
#                 torch.optim.Adam(
#                     self.app_module.embeds.parameters(),
#                     lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
#                     weight_decay=cfg.app_opt_reg,
#                 ),
#                 torch.optim.Adam(
#                     self.app_module.color_head.parameters(),
#                     lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
#                 ),
#             ]

#         # Losses & Metrics.
#         self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
#         self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
#         self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
#             self.device
#         )

#         # Viewer
#         if not self.cfg.disable_viewer:
#             self.server = viser.ViserServer(port=cfg.port, verbose=False)
#             self.viewer = GsplatViewer(
#                 server=self.server,
#                 render_fn=self._viewer_render_fn,
#                 output_dir=Path(cfg.result_dir),
#                 mode="training",
#             )

#     def rasterize_splats(
#         self,
#         camtoworlds: Tensor,
#         Ks: Tensor,
#         width: int,
#         height: int,
#         **kwargs,
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
#         means = self.splats["means"]  # [N, 3]
#         # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
#         # rasterization does normalization internally
#         quats = self.splats["quats"]  # [N, 4]
#         scales = torch.exp(self.splats["scales"])  # [N, 3]
#         opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        
#         image_ids = kwargs.pop("image_ids", None)

#         if self.cfg.app_opt:
#             colors = self.app_module(
#                 features=self.splats["features"],
#                 embed_ids=image_ids,
#                 dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
#                 sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
#             )
#             colors = colors + self.splats["colors"]
#             colors = torch.sigmoid(colors)
#         else:
#             colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

#         assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"
#         print(f"Colors {colors.shape} depths")
#         if self.model_type == "2dgs":
#             (
#                 render_colors,
#                 render_alphas,
#                 render_normals,
#                 normals_from_depth,
#                 render_distort,
#                 render_median,
#                 info,
#             ) = rasterization_2dgs(
#                 means=means,
#                 quats=quats,
#                 scales=scales,
#                 opacities=opacities,
#                 colors=colors,
#                 viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
#                 Ks=Ks,  # [C, 3, 3]
#                 width=width,
#                 height=height,
#                 packed=self.cfg.packed,
#                 absgrad=self.cfg.absgrad,
#                 sparse_grad=self.cfg.sparse_grad,
#                 **kwargs,
#             )
#         elif self.model_type == "2dgs-inria":
#             renders, info = rasterization_2dgs_inria_wrapper(
#                 means=means,
#                 quats=quats,
#                 scales=scales,
#                 opacities=opacities,
#                 colors=colors,
#                 viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
#                 Ks=Ks,  # [C, 3, 3]
#                 width=width,
#                 height=height,
#                 packed=self.cfg.packed,
#                 absgrad=self.cfg.absgrad,
#                 sparse_grad=self.cfg.sparse_grad,
#                 **kwargs,
#             )
#             render_colors, render_alphas = renders
#             render_normals = info["normals_rend"]
#             normals_from_depth = info["normals_surf"]
#             render_distort = info["render_distloss"]
#             render_median = render_colors[..., 3]

#         return (
#             render_colors,
#             render_alphas,
#             render_normals,
#             normals_from_depth,
#             render_distort,
#             render_median,
#             info,
#         )

#     def train(self):
#         cfg = self.cfg
#         device = self.device

#         # Dump cfg.
#         with open(f"{cfg.result_dir}/cfg.json", "w") as f:
#             json.dump(vars(cfg), f)

#         max_steps = cfg.max_steps
#         init_step = 0

#         schedulers = [
#             # means has a learning rate schedule, that end at 0.01 of the initial value
#             torch.optim.lr_scheduler.ExponentialLR(
#                 self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
#             ),
#         ]
#         if cfg.pose_opt:
#             # pose optimization has a learning rate schedule
#             schedulers.append(
#                 torch.optim.lr_scheduler.ExponentialLR(
#                     self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
#                 )
#             )

#         trainloader = torch.utils.data.DataLoader(
#             self.trainset,
#             batch_size=cfg.batch_size,
#             shuffle=True,
#             num_workers=4,
#             persistent_workers=True,
#             pin_memory=True,
#         )
#         trainloader_iter = iter(trainloader)

#         # Training loop.
#         global_tic = time.time()
#         pbar = tqdm.tqdm(range(init_step, max_steps))
#         for step in pbar:
#             if not cfg.disable_viewer:
#                 while self.viewer.state == "paused":
#                     time.sleep(0.01)
#                 self.viewer.lock.acquire()
#                 tic = time.time()

#             try:
#                 data = next(trainloader_iter)
#             except StopIteration:
#                 trainloader_iter = iter(trainloader)
#                 data = next(trainloader_iter)

#             camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
#             Ks = data["K"].to(device)  # [1, 3, 3]
#             pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
#             num_train_rays_per_step = (
#                 pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
#             )
#             image_ids = data["image_id"].to(device)
#             if cfg.depth_loss:
#                 points = data["points"].to(device)  # [1, M, 2]
#                 depths_gt = data["depths"].to(device)  # [1, M]

#             height, width = pixels.shape[1:3]

#             if cfg.pose_noise:
#                 camtoworlds = self.pose_perturb(camtoworlds, image_ids)

#             if cfg.pose_opt:
#                 camtoworlds = self.pose_adjust(camtoworlds, image_ids)

#             # sh schedule
#             sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

#             # forward
#             (
#                 renders,
#                 alphas,
#                 normals,
#                 normals_from_depth,
#                 render_distort,
#                 render_median,
#                 info,
#             ) = self.rasterize_splats(
#                 camtoworlds=camtoworlds,
#                 Ks=Ks,
#                 width=width,
#                 height=height,
#                 sh_degree=sh_degree_to_use,
#                 near_plane=cfg.near_plane,
#                 far_plane=cfg.far_plane,
#                 image_ids=image_ids,
#                 render_mode="RGB" if cfg.depth_loss else "RGB+D",
#                 distloss=self.cfg.dist_loss,
#             )
#             if renders.shape[-1] == 4:
#                 colors, depths = renders[..., 0:3], renders[..., 3:4]
#             else:
#                 colors, depths = renders, None

#             if cfg.random_bkgd:
#                 bkgd = torch.rand(1, 3, device=device)
#                 colors = colors + bkgd * (1.0 - alphas)

#             self.strategy.step_pre_backward(
#                 params=self.splats,
#                 optimizers=self.optimizers,
#                 state=self.strategy_state,
#                 step=step,
#                 info=info,
#             )
#             masks = data["mask"].to(device) if "mask" in data else None
#             if masks is not None:
#                 pixels = pixels * masks[..., None]
#                 colors = colors * masks[..., None]

#             # loss
#             l1loss = F.l1_loss(colors, pixels)
#             ssimloss = 1.0 - self.ssim(
#                 pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
#             )
#             loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)
#             if cfg.depth_loss:
#                 # query depths from depth map
#                 points = torch.stack(
#                     [
#                         points[:, :, 0] / (width - 1) * 2 - 1,
#                         points[:, :, 1] / (height - 1) * 2 - 1,
#                     ],
#                     dim=-1,
#                 )  # normalize to [-1, 1]
#                 grid = points.unsqueeze(2)  # [1, M, 1, 2]
#                 depths = F.grid_sample(
#                     depths.permute(0, 3, 1, 2), grid, align_corners=True
#                 )  # [1, 1, M, 1]
#                 depths = depths.squeeze(3).squeeze(1)  # [1, M]
#                 # calculate loss in disparity space
#                 disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
#                 disp_gt = 1.0 / depths_gt  # [1, M]
#                 depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
#                 loss += depthloss * cfg.depth_lambda

#             if cfg.normal_loss:
#                 if step > cfg.normal_start_iter:
#                     curr_normal_lambda = cfg.normal_lambda
#                 else:
#                     curr_normal_lambda = 0.0
#                 # normal consistency loss
#                 normals = normals.squeeze(0).permute((2, 0, 1))
#                 normals_from_depth *= alphas.squeeze(0).detach()
#                 if len(normals_from_depth.shape) == 4:
#                     normals_from_depth = normals_from_depth.squeeze(0)
#                 normals_from_depth = normals_from_depth.permute((2, 0, 1))
#                 normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
#                 normalloss = curr_normal_lambda * normal_error.mean()
#                 loss += normalloss

#             if cfg.dist_loss:
#                 if step > cfg.dist_start_iter:
#                     curr_dist_lambda = cfg.dist_lambda
#                 else:
#                     curr_dist_lambda = 0.0
#                 distloss = render_distort.mean()
#                 loss += distloss * curr_dist_lambda

#             loss.backward()

#             desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
#             if cfg.depth_loss:
#                 desc += f"depth loss={depthloss.item():.6f}| "
#             if cfg.dist_loss:
#                 desc += f"dist loss={distloss.item():.6f}"
#             if cfg.pose_opt and cfg.pose_noise:
#                 # monitor the pose error if we inject noise
#                 pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
#                 desc += f"pose err={pose_err.item():.6f}| "
#             pbar.set_description(desc)

#             if cfg.tb_every > 0 and step % cfg.tb_every == 0:
#                 mem = torch.cuda.max_memory_allocated() / 1024**3
#                 self.writer.add_scalar("train/loss", loss.item(), step)
#                 self.writer.add_scalar("train/l1loss", l1loss.item(), step)
#                 self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
#                 self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
#                 self.writer.add_scalar("train/mem", mem, step)
#                 if cfg.depth_loss:
#                     self.writer.add_scalar("train/depthloss", depthloss.item(), step)
#                 if cfg.normal_loss:
#                     self.writer.add_scalar("train/normalloss", normalloss.item(), step)
#                 if cfg.dist_loss:
#                     self.writer.add_scalar("train/distloss", distloss.item(), step)
#                 if cfg.tb_save_image:
#                     canvas = (
#                         torch.cat([pixels, colors[..., :3]], dim=2)
#                         .detach()
#                         .cpu()
#                         .numpy()
#                     )
#                     canvas = canvas.reshape(-1, *canvas.shape[2:])
#                     self.writer.add_image("train/render", canvas, step)
#                 self.writer.flush()

#             self.strategy.step_post_backward(
#                 params=self.splats,
#                 optimizers=self.optimizers,
#                 state=self.strategy_state,
#                 step=step,
#                 info=info,
#                 packed=cfg.packed,
#             )

#             # Turn Gradients into Sparse Tensor before running optimizer
#             if cfg.sparse_grad:
#                 assert cfg.packed, "Sparse gradients only work with packed mode."
#                 gaussian_ids = info["gaussian_ids"]
#                 for k in self.splats.keys():
#                     grad = self.splats[k].grad
#                     if grad is None or grad.is_sparse:
#                         continue
#                     self.splats[k].grad = torch.sparse_coo_tensor(
#                         indices=gaussian_ids[None],  # [1, nnz]
#                         values=grad[gaussian_ids],  # [nnz, ...]
#                         size=self.splats[k].size(),  # [N, ...]
#                         is_coalesced=len(Ks) == 1,
#                     )

#             # optimize
#             for optimizer in self.optimizers.values():
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
#             for optimizer in self.pose_optimizers:
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
#             for optimizer in self.app_optimizers:
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
#             for scheduler in schedulers:
#                 scheduler.step()

#             # save checkpoint
#             if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
#                 mem = torch.cuda.max_memory_allocated() / 1024**3
#                 stats = {
#                     "mem": mem,
#                     "ellipse_time": time.time() - global_tic,
#                     "num_GS": len(self.splats["means"]),
#                 }
#                 with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
#                     json.dump(stats, f)
#                 ckpt_data = {
#                     "step": step,
#                     "splats": self.splats.state_dict(),
#                     "n_train_images": len(self.trainset),
#                 }
#                 print("Step: ", step, stats)

#                 if cfg.app_opt:
#                     ckpt_data["app_module"] = self.app_module.state_dict()
#                     ckpt_data["app_embed_dim"] = cfg.app_embed_dim
#                     ckpt_data["feature_dim"] = cfg.feature_dim
                
                
#                 torch.save(
#                     ckpt_data,
#                     f"{self.ckpt_dir}/ckpt_{step}.pt",
#                 )

#             # eval the full set
#             if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
#                 self.eval(step)
#                 self.render_traj(step)

#             if not cfg.disable_viewer:
#                 self.viewer.lock.release()
#                 num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
#                 num_train_rays_per_sec = (
#                     num_train_rays_per_step * num_train_steps_per_sec
#                 )
#                 # Update the viewer state.
#                 self.viewer.render_tab_state.num_train_rays_per_sec = (
#                     num_train_rays_per_sec
#                 )
#                 # Update the scene.
#                 self.viewer.update(step, num_train_rays_per_step)

#     @torch.no_grad()
#     def eval(self, step: int):
#         """Entry for evaluation."""
#         print("Running evaluation...")
#         cfg = self.cfg
#         device = self.device

#         valloader = torch.utils.data.DataLoader(
#             self.valset, batch_size=1, shuffle=False, num_workers=1
#         )
#         ellipse_time = 0
#         metrics = {"psnr": [], "ssim": [], "lpips": []}
#         for i, data in enumerate(valloader):
#             camtoworlds = data["camtoworld"].to(device)
#             Ks = data["K"].to(device)
#             pixels = data["image"].to(device) / 255.0
#             height, width = pixels.shape[1:3]

#             torch.cuda.synchronize()
#             tic = time.time()
#             (
#                 colors,
#                 alphas,
#                 normals,
#                 normals_from_depth,
#                 render_distort,
#                 render_median,
#                 _,
#             ) = self.rasterize_splats(
#                 camtoworlds=camtoworlds,
#                 Ks=Ks,
#                 width=width,
#                 height=height,
#                 sh_degree=cfg.sh_degree,
#                 near_plane=cfg.near_plane,
#                 far_plane=cfg.far_plane,
#                 render_mode="RGB",
#             )  # [1, H, W, 3]
#             colors = torch.clamp(colors, 0.0, 1.0)
#             colors = colors[..., :3]  # Take RGB channels
#             torch.cuda.synchronize()
#             ellipse_time += max(time.time() - tic, 1e-10)

#             # write images
#             canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
#             imageio.imwrite(
#                 f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
#             )

#             # write median depths
#             render_median = (render_median - render_median.min()) / (
#                 render_median.max() - render_median.min()
#             )
#             # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
#             render_median = (
#                 apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
#             )

#             imageio.imwrite(
#                 f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
#                 (render_median * 255).astype(np.uint8),
#             )

#             # write normals
#             normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
#             normals_output = (normals * 255).astype(np.uint8)
#             imageio.imwrite(
#                 f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
#             )

#             # write normals from depth
#             normals_from_depth *= alphas.squeeze(0).detach()
#             normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
#             normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
#                 np.max(normals_from_depth) - np.min(normals_from_depth)
#             )
#             normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
#             if len(normals_from_depth_output.shape) == 4:
#                 normals_from_depth_output = normals_from_depth_output.squeeze(0)
#             imageio.imwrite(
#                 f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
#                 normals_from_depth_output,
#             )

#             # write distortions

#             render_dist = render_distort
#             dist_max = torch.max(render_dist)
#             dist_min = torch.min(render_dist)
#             render_dist = (render_dist - dist_min) / (dist_max - dist_min)
#             render_dist = (
#                 apply_float_colormap(render_dist).detach().cpu().squeeze(0).numpy()
#             )
#             imageio.imwrite(
#                 f"{self.render_dir}/val_{i:04d}_distortions_{step}.png",
#                 (render_dist * 255).astype(np.uint8),
#             )

#             pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
#             colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
#             metrics["psnr"].append(self.psnr(colors, pixels))
#             metrics["ssim"].append(self.ssim(colors, pixels))
#             metrics["lpips"].append(self.lpips(colors, pixels))

#         ellipse_time /= len(valloader)

#         psnr = torch.stack(metrics["psnr"]).mean()
#         ssim = torch.stack(metrics["ssim"]).mean()
#         lpips = torch.stack(metrics["lpips"]).mean()
#         print(
#             f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
#             f"Time: {ellipse_time:.3f}s/image "
#             f"Number of GS: {len(self.splats['means'])}"
#         )
#         # save stats as json
#         stats = {
#             "psnr": psnr.item(),
#             "ssim": ssim.item(),
#             "lpips": lpips.item(),
#             "ellipse_time": ellipse_time,
#             "num_GS": len(self.splats["means"]),
#         }
#         with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
#             json.dump(stats, f)
#         # save stats to tensorboard
#         for k, v in stats.items():
#             self.writer.add_scalar(f"val/{k}", v, step)
#         self.writer.flush()

#     @torch.no_grad()
#     def render_traj(self, step: int):
#         """Entry for trajectory rendering."""
#         print("Running trajectory rendering...")
#         cfg = self.cfg
#         device = self.device

#         camtoworlds = self.parser.camtoworlds[5:-5]
#         camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
#         camtoworlds = np.concatenate(
#             [
#                 camtoworlds,
#                 np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
#             ],
#             axis=1,
#         )  # [N, 4, 4]

#         camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
#         K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
#         width, height = list(self.parser.imsize_dict.values())[0]

#         canvas_all = []
#         for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
#             renders, _, _, surf_normals, _, _, _ = self.rasterize_splats(
#                 camtoworlds=camtoworlds[i : i + 1],
#                 Ks=K[None],
#                 width=width,
#                 height=height,
#                 sh_degree=cfg.sh_degree,
#                 near_plane=cfg.near_plane,
#                 far_plane=cfg.far_plane,
#                 render_mode="RGB",
#             )  # [1, H, W, 4]
#             colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
#             depths = renders[0, ..., 3:4]  # [H, W, 1]
#             depths = (depths - depths.min()) / (depths.max() - depths.min())

#             surf_normals = (surf_normals - surf_normals.min()) / (
#                 surf_normals.max() - surf_normals.min()
#             )
#             print(f"Shape colors and depeths {colors.shape} {depths.shape}")
#             # write images
#             canvas = torch.cat(
#                 [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
#             )
#             canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
#             canvas_all.append(canvas)

#         # save to video
#         video_dir = f"{cfg.result_dir}/videos"
#         os.makedirs(video_dir, exist_ok=True)
#         writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
#         for canvas in canvas_all:
#             writer.append_data(canvas)
#         writer.close()
#         print(f"Video saved to {video_dir}/traj_{step}.mp4")

#     @torch.no_grad()
#     def _viewer_render_fn(
#         self, camera_state: CameraState, render_tab_state: RenderTabState
#     ):
#         assert isinstance(render_tab_state, GsplatRenderTabState)
#         if render_tab_state.preview_render:
#             width = render_tab_state.render_width
#             height = render_tab_state.render_height
#         else:
#             width = render_tab_state.viewer_width
#             height = render_tab_state.viewer_height
#         c2w = camera_state.c2w
#         K = camera_state.get_K((width, height))
#         c2w = torch.from_numpy(c2w).float().to(self.device)
#         K = torch.from_numpy(K).float().to(self.device)

#         (
#             render_colors,
#             render_alphas,
#             render_normals,
#             normals_from_depth,
#             render_distort,
#             render_median,
#             info,
#         ) = self.rasterize_splats(
#             camtoworlds=c2w[None],
#             Ks=K[None],
#             width=width,
#             height=height,
#             sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
#             near_plane=render_tab_state.near_plane,
#             far_plane=render_tab_state.far_plane,
#             radius_clip=render_tab_state.radius_clip,
#             eps2d=render_tab_state.eps2d,
#             render_mode="RGB+ED",
#             backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
#             / 255.0,
#         )  # [1, H, W, 3]
#         render_tab_state.total_gs_count = len(self.splats["means"])
#         render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

#         if render_tab_state.render_mode == "depth":
#             # normalize depth to [0, 1]
#             depth = render_median
#             if render_tab_state.normalize_nearfar:
#                 near_plane = render_tab_state.near_plane
#                 far_plane = render_tab_state.far_plane
#             else:
#                 near_plane = depth.min()
#                 far_plane = depth.max()
#             depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
#             depth_norm = torch.clip(depth_norm, 0, 1)
#             if render_tab_state.inverse:
#                 depth_norm = 1 - depth_norm
#             renders = (
#                 apply_float_colormap(depth_norm, render_tab_state.colormap)
#                 .cpu()
#                 .numpy()
#             )
#         elif render_tab_state.render_mode == "normal":
#             render_normals = render_normals * 0.5 + 0.5  # normalize to [0, 1]
#             renders = render_normals.cpu().numpy()
#         elif render_tab_state.render_mode == "alpha":
#             alpha = render_alphas[0, ..., 0:1]
#             renders = (
#                 apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
#             )
#         else:
#             render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
#             renders = render_colors.cpu().numpy()
#         return renders


# def main(cfg: Config):
#     runner = Runner(cfg)

#     if cfg.ckpt is not None:
#         # run eval only
#         ckpt = torch.load(cfg.ckpt, map_location=runner.device)
#         for k in runner.splats.keys():
#             runner.splats[k].data = ckpt["splats"][k]
#         runner.eval(step=ckpt["step"])
#         runner.render_traj(step=ckpt["step"])
#     else:
#         runner.train()

#     if not cfg.disable_viewer:
#         print("Viewer running... Ctrl+C to exit.")
#         time.sleep(1000000)


# if __name__ == "__main__":
#     cfg = tyro.cli(Config)
#     cfg.adjust_steps(cfg.steps_scaler)
#     main(cfg)

"""
simple_trainer_2dgs_ground.py
==============================
2DGS trainer extended with:
  1. Ground-plane depth supervision (Dai et al. 2024 / Algorithm 1)
     — ground_plane_depth_guided() produces a per-pixel D_gt map from a
       binary ground segmentation mask and the Zw=0 plane assumption.
     — depth_supervision_loss() (Ldist) is added to the colour loss.

  2. Depth Anything V3 (DA3) monocular depth prior
     — DA3 gives a dense but scale-ambiguous depth map D_pre.
     — align_depth_scale() fits (scale, shift) against the sparse ground
       D_gt so that D_pre is converted to metric scale (D_reg).
     — D_reg is used as an extra dense supervision signal on non-ground
       pixels where D_gt = 0, weighted by DA3 confidence.

  3. Ground-mask loader
     — Expects PNG masks in  <data_dir>/ground_masks/<image_name>.png
       (white = ground, black = non-ground).
     — Falls back to an all-zero mask if the directory does not exist,
       so the trainer degrades gracefully to standard 2DGS.

Usage (same CLI as original trainer, new flags highlighted):

  python simple_trainer_2dgs_ground.py
      --data_dir /data/scene-1
      --result_dir /results/scene-1
      --ground_loss               # enable ground-plane Ldist
      --ground_lambda 2.3         # lambda (paper default)
      --da3_loss                  # enable DA3 dense prior
      --da3_lambda 0.5            # weight for DA3 term
      --da3_model DA3NESTED-GIANT-LARGE-1.1
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy
from nerfview import CameraState, RenderTabState, apply_float_colormap

# ── Ground-plane depth prior (imported from companion module) ──────────────
from ground_plane_guided import (
    ground_plane_depth_guided,
    depth_supervision_loss,
    align_depth_scale,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # ── viewer ────────────────────────────────────────────────────────────────
    disable_viewer: bool = False
    ckpt: Optional[str] = None

    # ── data ──────────────────────────────────────────────────────────────────
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/garden"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    port: int = 8080
    colmap_dir: str = ""

    # ── training schedule ─────────────────────────────────────────────────────
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # ── Gaussian initialisation ───────────────────────────────────────────────
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2

    # ── near / far ────────────────────────────────────────────────────────────
    near_plane: float = 0.2
    far_plane: float = 200.0

    # ── densification ─────────────────────────────────────────────────────────
    prune_opa: float = 0.05
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100

    # ── rasterisation ─────────────────────────────────────────────────────────
    packed: bool = False
    sparse_grad: bool = False
    absgrad: bool = False
    antialiased: bool = False
    revised_opacity: bool = False
    random_bkgd: bool = False

    # ── pose optimisation ─────────────────────────────────────────────────────
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    # ── appearance optimisation ───────────────────────────────────────────────
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6
    feature_dim: int = 32

    # ── standard depth loss (COLMAP sparse points) ────────────────────────────
    depth_loss: bool = False
    depth_lambda: float = 1e-2

    # ── normal / distortion ───────────────────────────────────────────────────
    normal_loss: bool = False
    normal_lambda: float = 5e-2
    normal_start_iter: int = 7_000
    dist_loss: bool = False
    dist_lambda: float = 1e-2
    dist_start_iter: int = 3_000

    # ── model type ────────────────────────────────────────────────────────────
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # ── tensorboard ───────────────────────────────────────────────────────────
    tb_every: int = 100
    tb_save_image: bool = False

    # =========================================================================
    # NEW: Ground-plane depth supervision (Algorithm 1, Dai et al. 2024)
    # =========================================================================
    ground_loss: bool = False
    """Enable Ldist ground-plane depth supervision."""

    ground_lambda: float = 2.3
    """λ weighting Ldist (paper default = 2.3)."""

    ground_mask_dir: str = ""
    """
    Directory containing per-image ground segmentation masks (PNG, white=ground).
    Defaults to <data_dir>/ground_masks if left empty.
    """

    # =========================================================================
    # NEW: Depth Anything V3 monocular prior
    # =========================================================================
    da3_loss: bool = False
    """Enable dense DA3 depth supervision on non-ground pixels."""

    da3_lambda: float = 0.5
    """Weight for the DA3 dense depth term."""

    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    """HuggingFace model ID for DA3."""

    da3_process_res: int = 756
    """Processing resolution (longer side) passed to DA3."""

    da3_conf_thresh_pct: float = 30.0
    """Percentile below which DA3 confidence is treated as unreliable."""

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


# =============================================================================
# GROUND MASK LOADER
# =============================================================================

def load_ground_masks(
    image_names: List[str],
    mask_dir: str,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load per-image ground masks from disk.  Returns [N, H, W] bool tensor.

    Expects <mask_dir>/<image_stem>.png  (white = ground, black = sky/field).
    Falls back to all-False if the directory or a specific file is missing.
    """
    from PIL import Image as PILImage

    masks = []
    mask_dir_path = Path(mask_dir)
    for name in image_names:
        stem = Path(name).stem
        candidates = [
            mask_dir_path / f"{stem}.png",
            mask_dir_path / f"{stem}.jpg",
            mask_dir_path / f"{name}",
        ]
        mask_tensor = torch.zeros(height, width, dtype=torch.bool, device=device)
        for p in candidates:
            if p.exists():
                img = PILImage.open(p).convert("L").resize(
                    (width, height), PILImage.NEAREST
                )
                mask_tensor = torch.from_numpy(
                    np.array(img) > 127
                ).to(device)
                break
        masks.append(mask_tensor)

    return torch.stack(masks, dim=0)  # [N, H, W]


# =============================================================================
# DA3 DEPTH PRIOR
# =============================================================================

class DA3DepthPrior:
    """
    Wraps Depth Anything V3 for offline or on-the-fly depth estimation.

    Preferred workflow: run DA3 once over the whole dataset (``precompute``)
    and cache the NPZ results.  During training, ``query`` loads the cached
    map for the current image batch without re-running the network.
    """

    def __init__(self, model_id: str, process_res: int, device: torch.device):
        self.model_id = model_id
        self.process_res = process_res
        self.device = device
        self._model = None          # lazy-loaded
        self._cache: Dict[str, np.ndarray] = {}
        self._conf_cache: Dict[str, np.ndarray] = {}

    def _load_model(self):
        if self._model is None:
            from depth_anything_3.api import DepthAnything3
            print(f"[DA3] Loading {self.model_id} …")
            self._model = DepthAnything3.from_pretrained(self.model_id)
            self._model = self._model.to(device=self.device)
            self._model.eval()
            print("[DA3] Model ready.")

    def precompute(self, image_paths: List[str], cache_npz: str):
        """
        Run DA3 over *all* images once, save to ``cache_npz``.
        Call this before starting training (e.g. in a setup script).
        """
        if Path(cache_npz).exists():
            print(f"[DA3] Cache found at {cache_npz}, skipping inference.")
            return

        self._load_model()
        import glob
        prediction = self._model.inference(
            sorted(image_paths),
            process_res=self.process_res,
        )
        np.savez_compressed(
            cache_npz,
            depth=prediction.depth.astype(np.float32),      # (N, H, W)
            conf=prediction.conf.astype(np.float32),         # (N, H, W)
            extrinsics=prediction.extrinsics.astype(np.float32),
            intrinsics=prediction.intrinsics.astype(np.float32),
            names=np.array(image_paths),
        )
        print(f"[DA3] Saved cache → {cache_npz}")

    def load_cache(self, cache_npz: str):
        """Load precomputed NPZ produced by ``precompute``."""
        data = np.load(cache_npz, allow_pickle=True)
        names = list(data["names"])
        for i, name in enumerate(names):
            key = Path(str(name)).stem
            self._cache[key] = data["depth"][i]   # (H, W) float32
            self._conf_cache[key] = data["conf"][i]

    def query(
        self,
        image_name: str,
        height: int,
        width: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (depth, confidence) tensors for a single image, or (None, None)
        if this image was not found in the cache.

        Both outputs are float32 tensors of shape [H, W] on CPU (will be
        moved to GPU by the caller).
        """
        key = Path(image_name).stem
        if key not in self._cache:
            return None, None

        depth = torch.from_numpy(
            self._resize_map(self._cache[key], height, width)
        ).float()
        conf = torch.from_numpy(
            self._resize_map(self._conf_cache[key], height, width)
        ).float()
        return depth, conf

    @staticmethod
    def _resize_map(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        if arr.shape == (h, w):
            return arr
        from PIL import Image as PILImage
        return np.array(
            PILImage.fromarray(arr).resize((w, h), PILImage.BILINEAR)
        )


# =============================================================================
# SPLAT CREATION  (unchanged from original)
# =============================================================================

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs   = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs   = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("init_type must be 'sfm' or 'random'")

    N = points.shape[0]
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg  = torch.sqrt(dist2_avg)
    scales    = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
    quats     = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    params = [
        ("means",     torch.nn.Parameter(points),     1.6e-4 * scene_scale),
        ("scales",    torch.nn.Parameter(scales),     5e-3),
        ("quats",     torch.nn.Parameter(quats),      1e-3),
        ("opacities", torch.nn.Parameter(opacities),  5e-2),
    ]

    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]),  2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]),  2.5e-3 / 20))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


# =============================================================================
# RUNNER
# =============================================================================

class Runner:
    """Engine for training and evaluation with ground-plane depth prior."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)
        self.cfg    = cfg
        self.device = "cuda"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir   = f"{cfg.result_dir}/ckpts";    os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir  = f"{cfg.result_dir}/stats";   os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders";  os.makedirs(self.render_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # ── Dataset ───────────────────────────────────────────────────────────
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            colmap_dir=cfg.colmap_dir,
        )
        self.trainset = Dataset(
            self.parser, split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"Scene scale: {self.scene_scale:.4f}")

        # ── Ground mask directory ─────────────────────────────────────────────
        mask_dir = cfg.ground_mask_dir or os.path.join(cfg.data_dir, "ground_masks")
        self.mask_dir = mask_dir
        if cfg.ground_loss and not Path(mask_dir).exists():
            print(
                f"[WARNING] ground_loss=True but mask dir '{mask_dir}' not found. "
                "Ground supervision will produce zero loss (no masked pixels)."
            )

        # ── DA3 depth prior ───────────────────────────────────────────────────
        self.da3: Optional[DA3DepthPrior] = None
        if cfg.da3_loss:
            self.da3 = DA3DepthPrior(cfg.da3_model, cfg.da3_process_res, self.device)
            da3_cache = os.path.join(cfg.result_dir, "da3_cache.npz")
            # Try loading a precomputed cache first
            if Path(da3_cache).exists():
                print(f"[DA3] Loading cache from {da3_cache}")
                self.da3.load_cache(da3_cache)
            else:
                # Precompute over training set then load
                train_image_paths = [
                    os.path.join(cfg.data_dir, n)
                    for n in self.parser.image_names
                ]
                self.da3.precompute(train_image_paths, da3_cache)
                self.da3.load_cache(da3_cache)

        # ── Gaussian model ────────────────────────────────────────────────────
        feature_dim = cfg.feature_dim if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print(f"Model initialised.  N_Gaussians = {len(self.splats['means'])}")
        self.model_type = cfg.model_type
        key_for_grad = "gradient_2dgs" if self.model_type == "2dgs" else "means2d"

        # ── Densification strategy ────────────────────────────────────────────
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_grad,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # ── Pose optimisation ─────────────────────────────────────────────────
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
        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        # ── Appearance optimisation ───────────────────────────────────────────
        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), cfg.feature_dim,
                cfg.app_embed_dim, cfg.sh_degree,
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

        # ── Metrics ───────────────────────────────────────────────────────────
        self.ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr  = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # ── Viewer ────────────────────────────────────────────────────────────
        if not cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    # =========================================================================
    # RASTERISATION  (unchanged API; ground loss computed separately)
    # =========================================================================

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:

        means     = self.splats["means"]
        quats     = self.splats["quats"]
        scales    = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        image_ids = kwargs.pop("image_ids", None)

        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = torch.sigmoid(colors + self.splats["colors"])
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        assert not self.cfg.antialiased, "Antialiased not supported for 2DGS"

        if self.model_type == "2dgs":
            (
                render_colors, render_alphas, render_normals,
                normals_from_depth, render_distort, render_median, info,
            ) = rasterization_2dgs(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks, width=width, height=height,
                packed=self.cfg.packed, absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad, **kwargs,
            )
        else:  # 2dgs-inria
            renders, info = rasterization_2dgs_inria_wrapper(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks, width=width, height=height,
                packed=self.cfg.packed, absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad, **kwargs,
            )
            render_colors, render_alphas = renders
            render_normals     = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort     = info["render_distloss"]
            render_median      = render_colors[..., 3]

        return (
            render_colors, render_alphas, render_normals,
            normals_from_depth, render_distort, render_median, info,
        )

    # =========================================================================
    # GROUND-PLANE DEPTH SUPERVISION  (Algorithm 1 integration)
    # =========================================================================

    def compute_ground_loss(
        self,
        camtoworlds: Tensor,   # [C, 4, 4]  cam-to-world
        Ks: Tensor,            # [C, 3, 3]
        depths_rendered: Tensor,  # [C, H, W]  rendered expected depth
        image_names: List[str],
        height: int,
        width: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Build the ground-plane D_gt map and return:
          - ground_loss  : scalar Ldist
          - D_gt         : [C, H, W] depth map (for logging / DA3 alignment)
        """
        # ── 1. Load ground masks for this batch ───────────────────────────────
        masks = load_ground_masks(
            image_names, self.mask_dir, height, width,
            device=torch.device(self.device),
        )  # [C, H, W] bool

        if not masks.any():
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            D_gt = torch.zeros(
                camtoworlds.shape[0], height, width,
                device=self.device, dtype=camtoworlds.dtype,
            )
            return zero, D_gt

        # ── 2. World-space Gaussian parameters ───────────────────────────────
        means  = self.splats["means"]                          # [N, 3]
        quats  = F.normalize(self.splats["quats"], dim=-1)     # [N, 4]
        scales = torch.exp(self.splats["scales"])              # [N, 3]

        from ground_plane_guided import quat_scale_to_covar
        covars = quat_scale_to_covar(quats, scales)           # [N, 3, 3]

        # viewmats = world-to-camera = inv(camtoworlds)
        viewmats = torch.linalg.inv(camtoworlds)              # [C, 4, 4]

        # ── 3. Run ground-plane pipeline (Algorithm 1) ────────────────────────
        _, _, _, _, D_gt, _ = ground_plane_depth_guided(
            means=means,
            covars=covars,
            viewmats=viewmats,
            Ks=Ks,
            masks=masks,
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        # D_gt: [C, H, W]  (0 where mask=False or degenerate geometry)

        # ── 4. Compute Ldist ─────────────────────────────────────────────────
        # depths_rendered is [C, H, W]; depth_supervision_loss expects [C, N]
        # (Gaussian-indexed).  We instead supervise pixel-wise directly:
        valid = D_gt > 0.0
        if not valid.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True), D_gt

        ground_loss = F.mse_loss(depths_rendered[valid], D_gt[valid])
        return ground_loss, D_gt

    # =========================================================================
    # DA3 DENSE DEPTH LOSS
    # =========================================================================

    def compute_da3_loss(
        self,
        image_names: List[str],
        depths_rendered: Tensor,    # [C, H, W]
        D_gt: Tensor,               # [C, H, W]  ground-plane map
        conf_thresh_pct: float,
    ) -> Tensor:
        """
        For pixels where D_gt = 0 (non-ground), use scale-aligned DA3 depth
        as a soft supervision signal.

        Steps:
          1. Load DA3 depth (D_pre) and confidence (conf) for each image.
          2. align_depth_scale() → metric D_reg using sparse D_gt as anchor.
          3. MSE between rendered depth and D_reg, weighted by conf.
        """
        C, H, W = depths_rendered.shape
        da3_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        n_valid_cams = 0

        for c, name in enumerate(image_names):
            D_pre_np, conf_np = self.da3.query(name, H, W)
            if D_pre_np is None:
                continue

            D_pre = D_pre_np.to(self.device)   # [H, W]
            conf  = conf_np.to(self.device)     # [H, W]

            # Confidence threshold — ignore the bottom ``conf_thresh_pct``%
            thresh = torch.quantile(conf, conf_thresh_pct / 100.0)
            reliable = conf > thresh             # [H, W] bool

            # align_depth_scale uses ground pixels (D_gt > 0) as anchors
            D_gt_c   = D_gt[c]                  # [H, W]
            D_pre_flat = D_pre.reshape(-1)
            D_sparse   = D_gt_c.reshape(-1)
            w_flat     = conf.reshape(-1)

            _, _, D_reg_flat = align_depth_scale(D_pre_flat, D_sparse, w_flat)
            D_reg = D_reg_flat.reshape(H, W)    # metric DA3 depth

            # Supervise non-ground, reliable pixels only
            non_ground = D_gt_c == 0.0
            mask = reliable & non_ground        # [H, W]

            if mask.any():
                pred = depths_rendered[c][mask]
                tgt  = D_reg[mask].detach()     # treat DA3 as fixed target
                w    = conf[mask]
                # Weighted MSE
                cam_loss = (w * (pred - tgt) ** 2).sum() / (w.sum() + 1e-8)
                da3_loss = da3_loss + cam_loss
                n_valid_cams += 1

        if n_valid_cams > 0:
            da3_loss = da3_loss / n_valid_cams

        return da3_loss

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    def train(self):
        cfg    = self.cfg
        device = self.device

        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f, indent=2)

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
            self.trainset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, persistent_workers=True, pin_memory=True,
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

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
            Ks       = data["K"].to(device)
            pixels   = data["image"].to(device) / 255.0
            image_ids   = data["image_id"].to(device)
            image_names = data.get("image_name", ["unknown"] * cfg.batch_size)

            height, width = pixels.shape[1:3]

            if cfg.depth_loss:
                points    = data["points"].to(device)
                depths_gt = data["depths"].to(device)

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # ── Forward rasterisation ─────────────────────────────────────────
            render_mode = "RGB+D"   # always get depth channel for ground loss
            (
                renders, alphas, normals, normals_from_depth,
                render_distort, render_median, info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks,
                width=width, height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode=render_mode,
                distloss=cfg.dist_loss,
            )

            if renders.shape[-1] == 4:
                colors, depths = renders[..., :3], renders[..., 3]   # [C,H,W,3], [C,H,W]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd   = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                params=self.splats, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=info,
            )

            masks_seg = data.get("mask")
            if masks_seg is not None:
                masks_seg = masks_seg.to(device)
                pixels = pixels * masks_seg[..., None]
                colors = colors * masks_seg[..., None]

            # ── Colour loss ───────────────────────────────────────────────────
            l1loss   = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

            # ── COLMAP sparse depth loss (optional) ───────────────────────────
            if cfg.depth_loss and depths is not None:
                pts_norm = torch.stack([
                    points[..., 0] / (width  - 1) * 2 - 1,
                    points[..., 1] / (height - 1) * 2 - 1,
                ], dim=-1)
                sampled = F.grid_sample(
                    depths.unsqueeze(1), pts_norm.unsqueeze(2),
                    align_corners=True,
                ).squeeze(3).squeeze(1)
                disp    = torch.where(sampled > 0, 1.0 / sampled, torch.zeros_like(sampled))
                disp_gt = 1.0 / depths_gt
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss      = loss + depthloss * cfg.depth_lambda

            # ── Ground-plane depth supervision (Ldist) ────────────────────────
            D_gt = None
            if cfg.ground_loss and depths is not None:
                ground_loss, D_gt = self.compute_ground_loss(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    depths_rendered=depths,   # [C, H, W]
                    image_names=list(image_names),
                    height=height,
                    width=width,
                )
                loss = loss + cfg.ground_lambda * ground_loss

            # ── DA3 dense depth prior ─────────────────────────────────────────
            if cfg.da3_loss and depths is not None and self.da3 is not None:
                if D_gt is None:
                    # No ground mask → use zero D_gt (align_depth_scale will
                    # skip alignment and DA3 depth is used as-is with no anchor)
                    D_gt = torch.zeros_like(depths)
                da3_loss = self.compute_da3_loss(
                    image_names=list(image_names),
                    depths_rendered=depths,
                    D_gt=D_gt,
                    conf_thresh_pct=cfg.da3_conf_thresh_pct,
                )
                loss = loss + cfg.da3_lambda * da3_loss

            # ── Normal consistency loss ───────────────────────────────────────
            if cfg.normal_loss:
                curr_nl = cfg.normal_lambda if step > cfg.normal_start_iter else 0.0
                n_sq = normals.squeeze(0).permute(2, 0, 1)
                nfd  = normals_from_depth * alphas.squeeze(0).detach()
                if nfd.dim() == 4:
                    nfd = nfd.squeeze(0)
                nfd = nfd.permute(2, 0, 1)
                normalloss = curr_nl * (1 - (n_sq * nfd).sum(dim=0)).mean()
                loss = loss + normalloss

            # ── Distortion loss ───────────────────────────────────────────────
            if cfg.dist_loss:
                curr_dl = cfg.dist_lambda if step > cfg.dist_start_iter else 0.0
                loss = loss + curr_dl * render_distort.mean()

            loss.backward()

            # ── Logging ───────────────────────────────────────────────────────
            desc = f"loss={loss.item():.3f} | sh={sh_degree_to_use}"
            if cfg.ground_loss:
                desc += f" | Ldist={ground_loss.item():.4f}"
            if cfg.da3_loss:
                desc += f" | Lda3={da3_loss.item():.4f}"
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.writer.add_scalar("train/loss",      loss.item(),     step)
                self.writer.add_scalar("train/l1loss",    l1loss.item(),   step)
                self.writer.add_scalar("train/ssimloss",  ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS",    len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem",       mem,             step)
                if cfg.ground_loss:
                    self.writer.add_scalar("train/ground_loss", ground_loss.item(), step)
                if cfg.da3_loss:
                    self.writer.add_scalar("train/da3_loss", da3_loss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach().cpu().numpy()
                    )
                    self.writer.add_image("train/render", canvas.reshape(-1, *canvas.shape[2:]), step)
                self.writer.flush()

            self.strategy.step_post_backward(
                params=self.splats, optimizers=self.optimizers,
                state=self.strategy_state, step=step,
                info=info, packed=cfg.packed,
            )

            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients require packed mode."
                gids = info["gaussian_ids"]
                for k in self.splats.keys():
                    g = self.splats[k].grad
                    if g is None or g.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gids[None],
                        values=g[gids],
                        size=self.splats[k].size(),
                        is_coalesced=(len(Ks) == 1),
                    )

            for opt in self.optimizers.values():
                opt.step(); opt.zero_grad(set_to_none=True)
            for opt in self.pose_optimizers:
                opt.step(); opt.zero_grad(set_to_none=True)
            for opt in self.app_optimizers:
                opt.step(); opt.zero_grad(set_to_none=True)
            for sched in schedulers:
                sched.step()

            # ── Checkpoint ───────────────────────────────────────────────────
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem   = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {"mem": mem, "ellipse_time": time.time() - global_tic,
                         "num_GS": len(self.splats["means"])}
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                ckpt_data = {"step": step, "splats": self.splats.state_dict(),
                             "n_train_images": len(self.trainset)}
                if cfg.app_opt:
                    ckpt_data.update(
                        app_module=self.app_module.state_dict(),
                        app_embed_dim=cfg.app_embed_dim,
                        feature_dim=cfg.feature_dim,
                    )
                torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_{step}.pt")

            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                nstps = 1.0 / max(time.time() - tic, 1e-10)
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    pixels.shape[0] * height * width * nstps
                )
                self.viewer.update(step, pixels.shape[0] * height * width)

    # =========================================================================
    # EVALUATION  (unchanged)
    # =========================================================================

    @torch.no_grad()
    def eval(self, step: int):
        print("Running evaluation …")
        cfg, device = self.cfg, self.device
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics: Dict[str, list] = {"psnr": [], "ssim": [], "lpips": []}

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks    = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            (colors, alphas, normals, normals_from_depth,
             render_distort, render_median, _) = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks,
                width=width, height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                render_mode="RGB",
            )
            colors = torch.clamp(colors[..., :3], 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(f"{self.render_dir}/val_{i:04d}.png",
                            (canvas * 255).astype(np.uint8))

            depth_vis = (render_median - render_median.min()) / (
                render_median.max() - render_median.min() + 1e-8)
            depth_vis = apply_float_colormap(depth_vis).detach().cpu().squeeze(0).numpy()
            imageio.imwrite(f"{self.render_dir}/val_{i:04d}_depth_{step}.png",
                            (depth_vis * 255).astype(np.uint8))

            pixels = pixels.permute(0, 3, 1, 2)
            colors = colors.permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)
        psnr  = torch.stack(metrics["psnr"]).mean()
        ssim  = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(f"PSNR={psnr:.3f}  SSIM={ssim:.4f}  LPIPS={lpips:.3f}  "
              f"t={ellipse_time:.3f}s/img  N={len(self.splats['means'])}")
        stats = {"psnr": psnr.item(), "ssim": ssim.item(), "lpips": lpips.item(),
                 "ellipse_time": ellipse_time, "num_GS": len(self.splats["means"])}
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        print("Rendering trajectory …")
        cfg, device = self.cfg, self.device
        c2ws = self.parser.camtoworlds[5:-5]
        c2ws = generate_interpolated_path(c2ws, 1)
        c2ws = np.concatenate([
            c2ws,
            np.repeat(np.array([[[0, 0, 0, 1]]]), len(c2ws), axis=0),
        ], axis=1)
        c2ws = torch.from_numpy(c2ws).float().to(device)
        K    = torch.from_numpy(
            list(self.parser.Ks_dict.values())[0]
        ).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        canvas_all = []
        for i in tqdm.trange(len(c2ws), desc="Traj"):
            renders, *_ = self.rasterize_splats(
                camtoworlds=c2ws[i:i+1], Ks=K[None],
                width=width, height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                render_mode="RGB",
            )
            colors = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
            canvas = (colors.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for c in canvas_all:
            writer.append_data(c)
        writer.close()

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        w = render_tab_state.render_width  if render_tab_state.preview_render else render_tab_state.viewer_width
        h = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K   = torch.from_numpy(camera_state.get_K((w, h))).float().to(self.device)
        (render_colors, render_alphas, render_normals, normals_from_depth,
         render_distort, render_median, info) = self.rasterize_splats(
            camtoworlds=c2w[None], Ks=K[None], width=w, height=h,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane, far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip, eps2d=render_tab_state.eps2d,
            render_mode="RGB+ED",
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
        )
        render_tab_state.total_gs_count    = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
        if render_tab_state.render_mode == "depth":
            depth = render_median
            near  = render_tab_state.near_plane if render_tab_state.normalize_nearfar else depth.min()
            far   = render_tab_state.far_plane  if render_tab_state.normalize_nearfar else depth.max()
            dn    = torch.clip((depth - near) / (far - near + 1e-10), 0, 1)
            if render_tab_state.inverse:
                dn = 1 - dn
            return apply_float_colormap(dn, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "normal":
            return (render_normals * 0.5 + 0.5).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            return apply_float_colormap(render_alphas[0, ..., :1], render_tab_state.colormap).cpu().numpy()
        else:
            return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(cfg: Config):
    runner = Runner(cfg)
    if cfg.ckpt is not None:
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running … Ctrl+C to exit.")
        time.sleep(1_000_000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)