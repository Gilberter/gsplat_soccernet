import os
from datasets.colmap import Dataset, Parser
import torch
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary


scene_1 = "/disk/SN-NVS-2026-raw/scene-1"
colmap_dir = "/disk/SN-NVS-2026-raw/scene-1/sparse/0"
ground_seg_dir="/disk/SN-NVS-2026-raw/scene-1/output_sam/z/field_masks"


cameras_path = os.path.join(colmap_dir,"cameras.bin")
cameras = read_cameras_binary(cameras_path)

points_path = os.path.join(colmap_dir,"points3D.bin")
points = read_points3D_binary(points_path)

images_path = os.path.join(colmap_dir,"images.bin")
images = read_images_binary(images_path)



def shape_str(t):
    if isinstance(t,torch.Tensor):
        return f"shape = {tuple(t.shape)}  dtype={t.dtype}"
    return str(type(t))

    
parser = Parser(
    data_dir=scene_1,
    factor=2,
    normalize=False,
    test_every=0,
    load_exposure=False,
    colmap_dir=colmap_dir
)

trainset = Dataset(
    parser,
    split="train",
    patch_size=None,
    load_depths=True, # Load Ground Truth Depths from COLMAP
    load_ground_masks=ground_seg_dir, # SAM Ground Segmentation Directory
)

trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

trainloader_iter = iter(trainloader)

try:
    data = next(trainloader_iter)
except StopIteration:
    trainloader_iter = iter(trainloader)
    data = next(trainloader_iter)


import imageio
import numpy as np
import torch

print("==== DATA KEYS ====")
for k in data:
    v = data[k]
    if isinstance(v, torch.Tensor):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    else:
        print(f"{k}: type={type(v)}")

print("\n==== SAVING DEBUG IMAGES ====")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

# -------- IMAGE --------
if "image" in data:
    img = to_numpy(data["image"])
    print("\n[IMAGE]")
    print("shape:", img.shape, "min:", img.min(), "max:", img.max())

    # Handle batch
    if img.shape[0] == 1:
        img = img.squeeze(axis=0)
        print("shape:", img.shape)
    # Handle CHW → HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # Normalize if needed
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    #imageio.imwrite("./logs/debug_image.png", img)

# -------- DEPTH --------
if "depth" in data:
    depth = to_numpy(data["depth"])
    print("\n[DEPTH]")
    print("shape:", depth.shape, "min:", depth.min(), "max:", depth.max())

    depth_vis = depth.copy()
    depth_vis = depth_vis - depth_vis.min()
    depth_vis = depth_vis / (depth_vis.max() + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    #imageio.imwrite("debug_depth.png", depth_vis)

# -------- GROUND MASK --------
if "ground_mask" in data:
    mask = to_numpy(data["ground_mask"])
    print("\n[GROUND MASK]")
    print("shape:", mask.shape, "unique:", np.unique(mask))

    mask_vis = (mask.astype(np.uint8) * 255)
    #imageio.imwrite("debug_mask.png", mask_vis)


