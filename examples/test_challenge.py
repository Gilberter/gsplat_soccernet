
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

#from fused_ssim import fused_ssim

import os

#import torch

from read_write_model import load_cameras_bin, read_images_binary, read_points3D_binary

# scene 1
scene_1_challenge = "/disk/SN-NVS-2026-raw/scene-1-challenge/renders/"

rendered_scene_1 = "/home/hensemberk/dev/SOCCER-NET/renders"

# device = "cuda"
# ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
# psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)


# load camera bins from scene_1

# scene_1 = "/disk/SN-NVS-2026-raw/scene-1/"

# sparse_scene_1 = os.path.join(scene_1,"sparse/0/")

# cameras_path = os.path.join(sparse_scene_1,"cameras.bin")
# imges_path = os.path.join(sparse_scene_1,"images.bin")

# cameras = load_cameras_bin(cameras_path)

# points_path = os.path.join(sparse_scene_1,"points3D.bin")
# points = read_points3D_binary(points_path)

# images_path = os.path.join(sparse_scene_1,"images.bin")
# images = read_images_binary(images_path)


vgg_dir = "/disk/SN-NVS-2026-raw/scene-1/vgg/sparse"
points_path = os.path.join(vgg_dir,"points3D.bin")
points = read_points3D_binary(points_path)

images_path = os.path.join(vgg_dir,"images.bin")
images = read_images_binary(images_path)

print(len(points))
print(len(images))

import os
import pycolmap
from pycolmap import SceneManager

vgg_dir = "/disk/SN-NVS-2026-raw/scene-1/vgg/sparse"
scene_1_dir = "/disk/SN-NVS-2026-raw/scene-1/sparse/0"

# 1. Check the files exist
required_files = ["cameras.bin", "images.bin", "points3D.bin"]
for f in required_files:
    path = os.path.join(scene_1_dir, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"{f}: exists={exists}, size={size} bytes")

# 2. Try loading with pycolmap's Reconstruction directly (more verbose errors)
try:
    reconstruction = pycolmap.Reconstruction(scene_1_dir)
    print(f"\npycolmap.Reconstruction loaded OK")
    print(f"  Cameras:  {len(reconstruction.cameras)}")
    print(f"  Images:   {len(reconstruction.images)}")
    print(f"  Points3D: {len(reconstruction.points3D)}")
except Exception as e:
    print(f"pycolmap.Reconstruction failed: {e}")

# 3. Try SceneManager
try:
    manager = SceneManager(scene_1_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    print(f"\nSceneManager loaded OK")
    print(f"  Cameras:  {len(manager.cameras)}")
    print(f"  Images:   {len(manager.images)}")
    print(f"  Points3D: {len(manager.points3D)}")
except Exception as e:
    print(f"SceneManager failed: {e}")

# 4. Sanity-check a single image entry -- does it have camera_id, qvec, tvec?
try:
    from read_write_model import read_images_binary
    images = read_images_binary(os.path.join(scene_1_dir, "images.bin"))
    sample_id, sample_img = next(iter(images.items()))
    print(f"\nSample image entry (id={sample_id}):")
    print(f"  name:      {sample_img.name}")
    print(f"  camera_id: {sample_img.camera_id}")
    print(f"  qvec:      {sample_img.qvec}")
    print(f"  tvec:      {sample_img.tvec}")
    print(f"  xys shape: {sample_img.xys.shape}")
    print(f"  point3D_ids shape: {sample_img.point3D_ids.shape}")
except Exception as e:
    print(f"Image sanity check failed: {e}")

import os
import struct
import numpy as np

vgg_dir = "/disk/SN-NVS-2026-raw/scene-1/vgg/sparse"

# ── 1. Fast count of points without loading everything ──────────────────────
def count_points3D_binary(path):
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
    return num_points

def count_images_binary(path):
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
    return num_images

def count_cameras_binary(path):
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
    return num_cameras

print("=== VGG sparse ===")
print(f"  cameras:  {count_cameras_binary(os.path.join(vgg_dir, 'cameras.bin'))}")
print(f"  images:   {count_images_binary(os.path.join(vgg_dir, 'images.bin'))}")
print(f"  points3D: {count_points3D_binary(os.path.join(vgg_dir, 'points3D.bin'))}")

# ── 2. Check cameras are well-formed (load just cameras, it's small) ─────────
from pycolmap import SceneManager
manager = SceneManager(vgg_dir)
manager.load_cameras()
manager.load_images()
print(f"\nSceneManager (no points) loaded OK")
print(f"  Cameras: {len(manager.cameras)}")
print(f"  Images:  {len(manager.images)}")

print(manager.cameras.items())
# ── 3. Inspect one camera ─────────────────────────────────────────────────────
cam_id, cam = next(iter(manager.cameras.items()))

print(f"\nSample camera (id={cam_id}):")
print(f"  model:  {cam.camera_type}")
print(f"  width:  {cam.width}")
print(f"  height: {cam.height}")

print(f"  params: {cam.fx, cam.fy, cam.cx, cam.cy}")

# ── 4. Inspect one image ──────────────────────────────────────────────────────
img_id, img = next(iter(manager.images.items()))
print(f"\nSample image (id={img_id}):")
print(f"  name:      {img.name}")
print(f"  camera_id: {img.camera_id}")
print(f"  qvec:      {img.qvec}")
print(f"  tvec:      {img.tvec}")

# ── 5. Check camera_ids in images all exist in cameras ───────────────────────
image_cam_ids = set(img.camera_id for img in manager.images.values())
camera_ids    = set(manager.cameras.keys())
missing = image_cam_ids - camera_ids
print(f"\nCamera IDs referenced by images but missing from cameras.bin: {missing}")
print("→ VGG output is compatible with SceneManager" if not missing else "→ MISMATCH — camera entries missing")

from datasets.colmap import Parser, Dataset

data_dir = "/disk/SN-NVS-2026-raw/scene-1/renders"
self.parser = Parser(
    data_dir=data_dir,
    factor=1,
    normalize=False,
    test_every=0,
    load_exposure=False,
    colmap_dir=vgg_dir
)

self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )

trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
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

# print(cameras[1])

# print(cameras[15])

# images = read_images_binary(imges_path)

# camera_id_wanted = 1
# filtered = {img_id: img for img_id, img in images.items() if img.camera_id == camera_id_wanted}
# print(filtered)


# from viewer_2dgs_render import load_challenges

# c2w_mats, ks_list, imsize_list, image_ids = load_challenges("/disk/SN-NVS-2026-raw/scene-1-challenge/sparse/0/",2)
# print(c2w_mats[0])
# c2w = torch.from_numpy(c2w_mats[0]).float().to(device)
# viewmat = c2w.inverse().unsqueeze(0)  # [1, 4, 4]
# print(viewmat)

# print(ks_list[0])


# CHECK DIMENTIONS OF THW IMAGES IN CHALLENGUE and the dimentions of the rendered for the challengue

# import imageio.v3 as iio



# scene_1_rendered = "/home/hensemberk/dev/SOCCER-NET/results/scene-1-challenge/renders/00000.png"

# scene_1_challenge = "/disk/SN-NVS-2026-raw/scene-1-challenge/renders/00000.png"


# img_1_rendered = iio.imread(scene_1_rendered) 

# img_1_challenge = iio.imread(scene_1_challenge) 

# print(img_1_rendered.shape)
# print(img_1_challenge.shape)