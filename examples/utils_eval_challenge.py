# The idea of this file is to have a viewer where it can switch between camera training views
# And download those exact views

# load the ckpt of the trained scene

# Camera State intrinsic parameters


import argparse
import math
import os
import collections

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from gsplat.rendering import rasterization
import imageio.v3 as iio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as qua2rot
from utils import AppearanceOptModule, AppearanceOptModuleV2, CameraOptModule, knn, rgb_to_sh, set_random_seed

import PIL

import cv2
def load_splats(ckpt_paths, device):
    means, quats, scales, opacities = [], [], [], []
    sh0, shN = [], []

    app_modules, features, colors_base = [],[],[] # WHEN TRAINED WHEN APPMODULE
    
    mode = None

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        splats = ckpt["splats"]
        
        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], p=2, dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))

        # Check which mode (app_opt or sh_opt)
        has_sh = "sh0" in splats and "shN" in splats
        has_app = "features" in splats and "colors" in splats
        
        if has_app and "app_module" in ckpt:
            n_train = ckpt.get("n_train_images")
            if n_train is None:
                raise ValueError(f"Checkpoint {ckpt_path} missing 'n_train_images'")
            
            embed_dim = ckpt.get("app_embed_dim", 32)
            feature_dim = ckpt.get("feature_dim", 32)

            features.append(splats["features"])

            base_colors = splats["colors"]
            base_colors_safe = torch.clamp(base_colors, min=-10, max=10)
            colors_base.append(base_colors)

            for key in splats.keys():
                print(f"  - {key}: {splats[key].shape}")

            if torch.isinf(base_colors).any() or torch.isnan(base_colors).any():
                print(f"⚠️  WARNING: Found inf/nan in base colors")
                print(f"   Before clamp: [{base_colors.min():.3f}, {base_colors.max():.3f}]")
                print(f"   After clamp:  [{base_colors_safe.min():.3f}, {base_colors_safe.max():.3f}]")


            app_module = AppearanceOptModuleV2(
                n=n_train,
                feature_dim=feature_dim,
                embed_dim=embed_dim,
                sh_degree=3,

            ).to(device)

            app_module.load_state_dict(ckpt["app_module"])
            app_module.eval()
            app_modules.append(app_module)


            print(f"App module type: {type(app_module)}")
            print(f"✓ App mode | features={splats['features'].shape} colors={splats['colors'].shape}")
            mode = "app"
        elif has_sh:
            # "Spherical Mode"
            sh0.append(splats["sh0"])
            shN.append(splats["shN"])
            print(f"✓ SH mode | sh0={splats['sh0'].shape} shN={splats['shN'].shape}")
            mode = "sh"
        else:
            raise ValueError(f"Unknown mode in {ckpt_path}")

    means     = torch.cat(means, dim=0)
    quats     = torch.cat(quats, dim=0)
    scales    = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    if mode == "app":
        features = torch.cat(features, dim=0)
        colors = torch.cat(colors_base, dim=0)
        sh_degree = None
        print(f"List App Modules {app_modules}")
        app_module = app_modules[0] 
        print(f"List 2 App Modules {app_module}")


        sh_degree = None
    if mode == "sh":
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]
        sh_degree = 3
        app_module = None
        features = None

    print(f"Loaded {len(means)} Gaussians | sh_degree={sh_degree}")
    return means, quats, scales, opacities, colors, sh_degree, app_modules[0] if mode == "app" else None, features if mode == "app" else None


@torch.no_grad()
def render_camera(means, quats, scales, opacities, colors, sh_degree,
                  c2w_np, K_np, width, height, near, far, device, app_module=None, features=None):
    c2w = torch.from_numpy(c2w_np).float().to(device)
    K   = torch.from_numpy(K_np).float().to(device)
    viewmat = c2w.inverse().unsqueeze(0)  # [1, 4, 4]
    K_batch = K.unsqueeze(0)              # [1, 3, 3]

    base_rgb = torch.sigmoid(colors)
    base_rgb = base_rgb[None]   
    if app_module is not None and features is not None:
        print(f"\n{'─'*80}")
        print(f"📊 APPEARANCE OPTIMIZATION PATH")
        print(f"{'─'*80}")
        
        # Check shapes match
        N_means = means.shape[0]
        N_features = features.shape[0]
        N_colors = colors.shape[0]
        
        print(f"Shape compatibility check:")
        print(f"  N (means):    {N_means}")
        print(f"  N (features): {N_features}")
        print(f"  N (colors):   {N_colors}")
        
        if N_features != N_colors:
            print(f"❌ ERROR: features ({N_features}) != colors ({N_colors})")
            print(f"   This usually means multiple checkpoints were concatenated")
            print(f"   App module can only be used with single checkpoint!")
            raise RuntimeError(
                f"Appearance module shape mismatch: "
                f"features={N_features} but colors={N_colors}"
            )
        
        # Compute view directions
        dirs = means[None, :, :] - c2w[None, :3, 3]  # [1, N, 3]
        print(f"\nView directions:")
        print(f"  dirs shape:  {dirs.shape}")
        print(f"  dirs range:  [{dirs.min():.3f}, {dirs.max():.3f}]")
        
        # Get color delta from network
        print(f"\nCalling app_module...")
        print(f"  features:    {features.shape}")
        print(f"  embed_ids:   None (zero embedding)")
        print(f"  dirs:        {dirs.shape}")
        print(f"  sh_degree:   3")
        
        color_delta = app_module(
            features=features,
            embed_ids=None,      # Zero embedding for neutral view
            dirs=dirs,
            sh_degree=3,
            base_colors=base_rgb
        )  # Expected: [1, N, 3]
        
        print(f"\nNetwork output:")
        print(f"  color_delta shape:  {color_delta.shape}")
        print(f"  color_delta range:  [{color_delta.min():.3f}, {color_delta.max():.3f}]")
        print(f"  color_delta finite: {torch.isfinite(color_delta).all()}")
        
        # # ✅ Clamp base colors to prevent inf/nan
        # colors_safe = torch.clamp(colors, min=-100, max=100)
        
        # print(f"\nBase colors (logit space):")
        # print(f"  colors orig shape:   {colors.shape}")
        # print(f"  colors orig range:   [{colors.min():.3f}, {colors.max():.3f}]")
        # print(f"  colors has inf:      {torch.isinf(colors).any()}")
        # print(f"  colors has nan:      {torch.isnan(colors).any()}")
        # print(f"  colors_safe range:   [{colors_safe.min():.3f}, {colors_safe.max():.3f}]")
        
        # # ✅ Squeeze color_delta to match colors shape
        # color_delta_squeezed = color_delta.squeeze(0)  # [1, N, 3] → [N, 3]
        
        # print(f"\nColor delta after squeeze:")
        # print(f"  shape:  {color_delta_squeezed.shape}")
        # print(f"  range:  [{color_delta_squeezed.min():.3f}, {color_delta_squeezed.max():.3f}]")
        
        # # ✅ Combine: delta + base color (logit space)
        # print(f"\nCombining colors:")
        # # print(f"  color_delta_squeezed:  {color_delta_squeezed.shape}")
        # print(f"  colors_safe:           {colors_safe.shape}")
        
        # colors_combined = colors_safe + color_delta_squeezed 
        
        # print(f"  colors_combined (before clamp):")
        # print(f"    shape:   {colors_combined.shape}")
        # print(f"    range:   [{colors_combined.min():.3f}, {colors_combined.max():.3f}]")
        # print(f"    has inf: {torch.isinf(colors_combined).any()}")
        # print(f"    has nan: {torch.isnan(colors_combined).any()}")
        
        # # ✅ Clamp to prevent extreme values
        # colors_combined = torch.clamp(colors_combined, min=-100, max=100)
        
        # print(f"  colors_combined (after clamp):")
        # print(f"    range:   [{colors_combined.min():.3f}, {colors_combined.max():.3f}]")
        
        # # ✅ Apply sigmoid
        # colors_final = torch.sigmoid(colors_combined)

        # # ✅ Replace any NaN with neutral gray
        # nan_mask = ~torch.isfinite(colors_final)
        # if nan_mask.any():
        #     num_nans = nan_mask.sum().item()
        #     print(f"⚠️  WARNING: Found {num_nans} NaN/Inf values in final colors")
        #     colors_final[nan_mask] = 0.5  # Neutral gray
        #     print(f"   Replaced with 0.5 (neutral gray)")

        colors_final = color_delta.squeeze(0)
        render_sh_degree = None
    else:
        colors_final  = colors
        print(f"  colors orig shape:   {colors.shape}")
        print(f"SH DEGREE{sh_degree}")
        print(f"    range SH:   [{colors.min():.3f}, {colors.max():.3f}]")

    render_out, render_alphas, info = rasterization(
        means, quats, scales, opacities, colors_final,
        viewmat, K_batch,
        width=width, height=height,
        sh_degree=sh_degree,
        near_plane=near,
        far_plane=far,
        render_mode="RGB+ED",
        packed=False,
    )
    n_rendered = (info["radii"] > 0).all(-1).sum().item()
    print(f"    range n_rendered:   [{render_out.min():.3f}, {render_out.max():.3f}]")
    render_out[..., :3] = render_out[..., :3].clamp(0, 1)
    return render_out.cpu().numpy(), n_rendered


def save_outputs(output_dir, tag, rgb):

    os.makedirs(f"{output_dir}/renders_ours/", exist_ok=True)
    imageio.imwrite(
        f"{output_dir}/renders_ours/{tag}.png",
        (rgb * 255).clip(0, 255).astype(np.uint8)
    )

def save_outputs_canvas(output_dir, left_image, right_image, tag):

    # Add label "Ours"
    cv2.putText(
        left_image,
        "Ours",
        (20*5, 40*5),                     # position (x,y)
        cv2.FONT_HERSHEY_SIMPLEX,
        5,                          # font scale
        (255, 255, 255),              # white text
        2,                            # thickness
        cv2.LINE_AA
    )

    os.makedirs(f"{output_dir}/render_compared", exist_ok=True)

    print("left:", left_image.shape, left_image.dtype, left_image.min(), left_image.max())
    print("right:", right_image.shape, right_image.dtype)

    # ---- convert right image to float [0,1] ----
    if right_image.dtype == np.uint8:
        right_image = right_image.astype(np.float32) / 255.0

    # ---- ensure left image is float ----
    if left_image.dtype != np.float32:
        left_image = left_image.astype(np.float32)

    # ---- resize left if needed ----
    if left_image.shape[0] / 2 == right_image.shape[0]:
        resized_img = cv2.resize(left_image, None, fx=0.5, fy=0.5)
    else:
        resized_img = left_image

    print("resized:", resized_img.shape)

    # ---- match heights ----
    if resized_img.shape[0] != right_image.shape[0]:
        right_image = cv2.resize(
            right_image,
            (right_image.shape[1], resized_img.shape[0])
        )

    # ---- concatenate ----
    canvas = np.concatenate((resized_img, right_image), axis=1)

    # ---- save ----
    imageio.imwrite(
        f"{output_dir}/render_compared/{tag}.png",
        (canvas * 255).clip(0,255).astype(np.uint8)
    )
    
# this functions loads a txt from scenes challengues and gives
#   Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

def load_challenges(path, factor = 2):

    images = {}
    cameras = {}
    images_path = os.path.join(path,"images.txt")
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
                elems = fid.readline().split()
          
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name)

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
                    id=camera_id, model=model,
                    width=width, height=height,
                    params=params)
                    
    image_ids  = sorted(images.keys())
    c2w_mats = list()
    ks_list = list()       
    imsize_list = list()  # width, height 
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for image_id in image_ids:
        image_object = images[image_id]
        quat = qua2rot.from_quat(image_object.qvec,scalar_first=True)
        rot = quat.as_matrix()
        trans = image_object.tvec

        
        #world to camera position
        camera_position = -quat.inv().apply(trans)

        extrinsic_matrix = np.hstack((rot, trans.reshape(3,1)))

        w2c = np.vstack((extrinsic_matrix,bottom))

        c2w = np.linalg.inv(w2c)
        c2w_mats.append(c2w)

        # camera intrinsic
        camera_id = image_object.camera_id
        camera = cameras[camera_id]
        fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= factor
        
        # store per images
        ks_list.append(K)
        
        imsize_list.append((camera.width // factor, camera.height // factor))
    
    c2w_mats = np.stack(c2w_mats, axis=0)  # [N, 4, 4]
        
    return c2w_mats, ks_list, imsize_list, image_ids

def delete_image(output_dir, tag):

    paths = [
        f"{output_dir}/renders/{tag}.png",
        f"{output_dir}/renders_ours/{tag}.png",
        f"{output_dir}/render_compared/{tag}.png"
    ]

    for p in paths:
        if os.path.exists(p):
            os.remove(p)

import os
import glob

output_dir = "/disk/SN-NVS-2026-raw/scene-1-challenge"

def rename_tags_6_to_5(output_dir):

    folders = [
        f"{output_dir}/renders_ours",
        f"{output_dir}/render_compared"
    ]

    for folder in folders:

        if not os.path.exists(folder):
            continue

        for path in glob.glob(f"{folder}/*.png"):

            filename = os.path.basename(path)
            number = int(filename.split(".")[0])  # 000123 -> 123

            new_tag = f"{number:05d}"
            new_path = os.path.join(folder, f"{new_tag}.png")

            os.rename(path, new_path)

            print(f"{filename} -> {new_tag}.png")



