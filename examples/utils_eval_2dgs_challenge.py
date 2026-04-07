
import argparse
import math
import os
import collections

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from gsplat.rendering import rasterization_2dgs
from scipy.spatial.transform import Rotation as qua2rot
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

import PIL
import cv2


def load_splats(ckpt_paths, device, model_type="2dgs"):
    """Load 2DGS checkpoints - supports both SH and appearance optimization modes."""
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    app_modules, features, colors_base = [], [], []
    mode = None

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        splats = ckpt["splats"]

        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], p=2, dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))

        # ========== 2DGS-specific: Check for appearance optimization ==========
        if "sh0" not in splats or "shN" not in splats:
            # Appearance optimization mode
            n_train = ckpt.get("n_train_images")
            if n_train is None:
                raise ValueError(
                    f"Checkpoint {ckpt_path} has no 'n_train_images' key. "
                    "Pass --n_train_images explicitly or re-save the checkpoint."
                )
            print(f"  ✓ Appearance mode detected | n_train_images={n_train}")

            embed_dim = ckpt.get("app_embed_dim", 16)
            features.append(splats["features"])  # [N, feature_dim]
            colors_base.append(splats["colors"])  # [N, 3] logit-space base colors

            for key in splats.keys():
                print(f"    - {key}: {splats[key].shape}")

            # Load appearance module
            app_module = AppearanceOptModule(
                n=n_train,
                feature_dim=32,  # Must match training!
                embed_dim=embed_dim,  # Must match training!
                sh_degree=3,  # Must match training!
            ).to(device)

            if "app_module" in ckpt:
                app_module.load_state_dict(ckpt["app_module"])
                print(f"  ✓ App module state_dict loaded")
            else:
                print(f"  ⚠️  No app_module in checkpoint, using fresh initialization")

            app_module.eval()  # Set to evaluation mode
            app_modules.append(app_module)
            mode = "app"

        else:
            # SH mode (no appearance optimization)
            sh0.append(splats["sh0"])
            shN.append(splats["shN"])
            print(f"  ✓ SH mode | sh0={splats['sh0'].shape} shN={splats['shN'].shape}")
            mode = "sh"

    # Concatenate all loaded gaussians
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)

    if mode == "app":
        features = torch.cat(features, dim=0)
        colors = torch.cat(colors_base, dim=0)
        sh_degree = None
        print(
            f"✓ Loaded {len(means)} Gaussians in APPEARANCE mode | sh_degree={sh_degree}"
        )
    elif mode == "sh":
        colors = torch.cat([torch.cat(sh0, dim=0), torch.cat(shN, dim=0)], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        app_modules = [None]
        features = None
        print(
            f"✓ Loaded {len(means)} Gaussians in SH mode | sh_degree={sh_degree}"
        )
    else:
        raise ValueError("No splats loaded!")

    return (
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree,
        app_modules[0] if mode == "app" else None,
        features if mode == "app" else None,
    )


@torch.no_grad()
def render_camera_2dgs(
    means,
    quats,
    scales,
    opacities,
    colors,
    sh_degree,
    c2w_np,
    K_np,
    width,
    height,
    near,
    far,
    device,
    app_module=None,
    features=None,
):
    """Render a single camera view using 2DGS rasterization."""
    c2w = torch.from_numpy(c2w_np).float().to(device)
    K = torch.from_numpy(K_np).float().to(device)

    # Prepare camera matrices for 2DGS
    viewmat = torch.linalg.inv(c2w).unsqueeze(0)  # [1, 4, 4]
    K_batch = K.unsqueeze(0)  # [1, 3, 3]

    # ========== Prepare colors ==========
    if app_module is not None and features is not None:
        # ── Appearance mode: MLP-based color prediction ──
        # Compute view directions for appearance network
        dirs = means[None, :, :] - c2w[None, :3, 3]  # [1, N, 3]

        # Get appearance-dependent colors
        render_colors_eval = app_module(
            features=features,
            embed_ids=None,  # None → zero embedding (novel / neutral view)
            dirs=dirs,
            sh_degree=3,
        )
        # Add residual colors
        render_colors_eval = render_colors_eval + colors  # [1, N, 3]
        render_colors_eval = torch.sigmoid(render_colors_eval)  # [1, N, 3]
        print(f"    Appearance mode: computed colors with MLP")
    else:
        # ── SH mode: use spherical harmonics ──
        render_colors_eval = colors  # Already [N, K, 3]
        print(f"    SH mode: using SH coefficients (sh_degree={sh_degree})")

    # ========== 2DGS Rasterization ==========
    (
        render_out,
        render_alphas,
        render_normals,
        normals_from_depth,
        render_distort,
        render_median,
        info,
    ) = rasterization_2dgs(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=render_colors_eval,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K_batch,  # [1, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        near_plane=near,
        far_plane=far,
        render_mode="RGB+ED",  # RGB + Expected Depth
        packed=False,
    )

    # Get number of rendered gaussians
    n_rendered = (info["radii"] > 0).all(-1).sum().item()

    # Clamp colors to valid range
    render_out[..., :3] = render_out[..., :3].clamp(0, 1)

    return render_out.cpu().numpy(), n_rendered


def save_outputs(output_dir, tag, rgb):
    """Save rendered RGB output."""
    os.makedirs(f"{output_dir}/renders_ours/", exist_ok=True)
    imageio.imwrite(
        f"{output_dir}/renders_ours/{tag}.png",
        (rgb * 255).clip(0, 255).astype(np.uint8),
    )


def save_outputs_canvas(output_dir, left_image, right_image, tag):
    """Save side-by-side comparison of ours vs reference."""
    # ──── Add label "Ours" ────
    left_image_copy = left_image.copy()
    cv2.putText(
        left_image_copy,
        "Ours",
        (20 * 5, 40 * 5),  # position (x,y)
        cv2.FONT_HERSHEY_SIMPLEX,
        5,  # font scale
        (255, 255, 255),  # white text
        2,  # thickness
        cv2.LINE_AA,
    )

    os.makedirs(f"{output_dir}/render_compared", exist_ok=True)

    print(
        f"    left: {left_image.shape} {left_image.dtype} "
        f"[{left_image.min():.3f}, {left_image.max():.3f}]"
    )
    print(
        f"    right: {right_image.shape} {right_image.dtype} "
        f"[{right_image.min()}, {right_image.max()}]"
    )

    # ──── Convert right image to float [0,1] ────
    if right_image.dtype == np.uint8:
        right_image = right_image.astype(np.float32) / 255.0

    # ──── Ensure left image is float ────
    if left_image_copy.dtype != np.float32:
        left_image_copy = left_image_copy.astype(np.float32)

    # ──── Resize left if needed ────
    if left_image_copy.shape[0] / 2 == right_image.shape[0]:
        resized_img = cv2.resize(left_image_copy, None, fx=0.5, fy=0.5)
    else:
        resized_img = left_image_copy

    print(f"    resized: {resized_img.shape}")

    # ──── Match heights ────
    if resized_img.shape[0] != right_image.shape[0]:
        right_image = cv2.resize(
            right_image, (right_image.shape[1], resized_img.shape[0])
        )

    # ──── Concatenate ────
    canvas = np.concatenate((resized_img, right_image), axis=1)

    # ──── Save ────
    imageio.imwrite(
        f"{output_dir}/render_compared/{tag}.png",
        (canvas * 255).clip(0, 255).astype(np.uint8),
    )


# ============================================================
# COLMAP loading utilities
# ============================================================

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name"]
)

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)


def load_challenges(path, factor=2):
    """Load camera poses and intrinsics from COLMAP sparse format."""
    images = {}
    cameras = {}
    images_path = os.path.join(path, "images.txt")
    cameras_path = os.path.join(path, "cameras.txt")

    with open(images_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                # Skip points2D line
                fid.readline()

                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                )

    with open(cameras_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )

    image_ids = sorted(images.keys())
    c2w_mats = []
    ks_list = []
    imsize_list = []  # (width, height) per image
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    for image_id in image_ids:
        image_object = images[image_id]
        quat = qua2rot.from_quat(image_object.qvec, scalar_first=True)
        rot = quat.as_matrix()
        trans = image_object.tvec

        # World to camera position
        camera_position = -quat.inv().apply(trans)

        extrinsic_matrix = np.hstack((rot, trans.reshape(3, 1)))
        w2c = np.vstack((extrinsic_matrix, bottom))
        c2w = np.linalg.inv(w2c)
        c2w_mats.append(c2w)

        # Camera intrinsic
        camera_id = image_object.camera_id
        camera = cameras[camera_id]
        fx, fy, cx, cy = (
            camera.params[0],
            camera.params[1],
            camera.params[2],
            camera.params[3],
        )
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= factor

        # Store per image
        ks_list.append(K)
        imsize_list.append((camera.width // factor, camera.height // factor))

    c2w_mats = np.stack(c2w_mats, axis=0)  # [N, 4, 4]

    return c2w_mats, ks_list, imsize_list, image_ids