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

from scipy.spatial.transform import Rotation as qua2rot
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

import PIL

import cv2
def load_splats(ckpt_paths, device):
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    app_modules,features,colors_base = [],[],[] # WHEN TRAINED WHEN APPMODULE
    mode = None
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        splats = ckpt["splats"]
        
        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], p=2, dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))
        
      
        if "sh0" not in splats or "shN" not in splats:
            n_train = ckpt.get("n_train_images")
            if n_train is None:
                raise ValueError(
                    f"Checkpoint {ckpt_path} has no 'n_train_images' key. "
                    "Pass --n_train_images explicitly or re-save the checkpoint."
                )
            print(f"  n_train_images={n_train}")
            
            embed_dim = ckpt.get("app_embed_dim")
            feature_dim = ckpt.get("feature_dim")
            features.append(splats["features"])   # [N, 32]
            colors_base.append(splats["colors"])  # [N, 3] logit-space base colors
            for key in splats.keys():
                print(f"  - {key}: {splats[key].shape}")
            app_state = ckpt["app_module"] if "app_module" in ckpt else None
            print(f"Warning: 'sh0' or 'shN' not found in {ckpt_path}. Using RGB colors instead.")
            app_module = AppearanceOptModule(
                n=n_train,     # Must match training!
                feature_dim=feature_dim,          # Must match training!
                embed_dim=embed_dim,     # Must match training!
                sh_degree=3,             # Must match training!
            ).to(device)
            app_module.load_state_dict(ckpt["app_module"])
            print(f"✓ Found app_module state_dict with keys")
            print(f"  → App mode | features={splats['features'].shape} {splats['colors'].shape} | app_module loaded ✓")

            app_module.eval()  # Set to evaluation mode
            app_modules.append(app_module)
            mode = "app"
        else:
            
            sh0.append(splats["sh0"])
            shN.append(splats["shN"])
            print(f"  → SH mode | sh0={splats['sh0'].shape} shN={splats['shN'].shape}")

            mode = "sh"
    means     = torch.cat(means, dim=0)
    quats     = torch.cat(quats, dim=0)
    scales    = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    if mode == "app":
        features  = torch.cat(features, dim=0)
        colors = torch.cat(colors_base, dim=0)
        sh_degree = None
    if mode == "sh":
        colors    = torch.cat([torch.cat(sh0, dim=0), torch.cat(shN, dim=0)], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print(f"Loaded {len(means)} Gaussians | sh_degree={sh_degree}")
    return means, quats, scales, opacities, colors, sh_degree, app_modules[0] if mode == "app" else None, features if mode == "app" else None


@torch.no_grad()
def render_camera(means, quats, scales, opacities, colors, sh_degree,
                  c2w_np, K_np, width, height, near, far, device, app_module=None, features=None):
    c2w = torch.from_numpy(c2w_np).float().to(device)
    K   = torch.from_numpy(K_np).float().to(device)
    viewmat = c2w.inverse().unsqueeze(0)  # [1, 4, 4]
    K_batch = K.unsqueeze(0)              # [1, 3, 3]

    if app_module is not None and features is not None:
        # ── App mode: evaluate MLP with neutral (zero) embedding ──────────
        dirs = means[None, :, :] - c2w[None, :3, 3]   # [1, N, 3]
        render_colors_eval = app_module(
            features  = features,
            embed_ids = None,      # None → zero embedding (novel / neutral view)
            dirs      = dirs,
            sh_degree = 3,
        )
        render_colors_eval = render_colors_eval + colors   # residual
        render_colors_eval = torch.sigmoid(render_colors_eval)  # [1, N, 3]
    else:
        render_colors_eval = colors
    render_out, render_alphas, info = rasterization(
        means, quats, scales, opacities, render_colors_eval,
        viewmat, K_batch,
        width=width, height=height,
        sh_degree=sh_degree,
        near_plane=near,
        far_plane=far,
        render_mode="RGB+ED",
        packed=False,
    )
    n_rendered = (info["radii"] > 0).all(-1).sum().item()

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



