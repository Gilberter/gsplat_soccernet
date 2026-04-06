"""
test_da3_depth.py
=================
Validates DA3 NPZ output:
  1. Checks depth/conf/extrinsics/intrinsics shapes and ranges
  2. Compares DA3 intrinsics/extrinsics against COLMAP ground truth
  3. Back-projects depth → 3D point cloud and saves .ply
  4. Saves left-right canvas: rendered depth heatmap | original RGB image
"""

import os
import sys
import glob
import struct
import torch
import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image

from pycolmap import SceneManager



# ── optional: suppress Open3D verbose output ──────────────────────────────────
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

sys.path.append('/home/hensemberk/dev/Soccernet')
# =============================================================================
# CONFIGURATION — edit these paths
# =============================================================================
NPZ_PATH    = "/disk/SN-NVS-2026-raw/scene-1/scene-1-dae3/depth_maps.npz"
SPARSE_DIR  = "/disk/SN-NVS-2026-raw/scene-1/sparse/0"
IMAGES_DIR  = "/disk/SN-NVS-2026-raw/scene-1/images"   # folder with the RGB images
OUTPUT_DIR  = "./da3_test_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/render_compared", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/pointclouds",     exist_ok=True)

manager = SceneManager(SPARSE_DIR)
manager.load_cameras()
manager.load_images()
print(f"\nSceneManager (no points) loaded OK")
print(f"  Cameras: {len(manager.cameras)}")
print(f"  Images:  {len(manager.images)}")

imdata = manager.images
extrinsics = []
intrinsics = []
for k in imdata:

    # EXTRINSINCS
    
    im = imdata[k]
    rot = im.R()
    trans = im.tvec.reshape(3, 1)
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
    extrinsics.append(w2c)

    # INTRINSICS

    camera_id = im.camera_id    
    cam = manager.cameras[camera_id]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= 2
    intrinsics.append(K)

intrinsics = np.stack(intrinsics,axis=0)
extrinsics = np.stack(extrinsics,axis=0)

print(f"Intrinsic shape {intrinsics.shape}")
print(f"extrinsics shape {extrinsics.shape}")

# =============================================================================
# HELPERS
# =============================================================================

def depth_to_3d(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Backproject a depth map to 3-D world points.
    depth : (H, W)  metric depth in metres
    K     : (3, 3)  intrinsics
    c2w   : (4, 4)  camera-to-world  (i.e. inverse of world-to-camera / extrinsic)
    Returns (H*W, 3) world-space points.
    """
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    x = (xx - K[0, 2]) * depth / K[0, 0]
    y = (yy - K[1, 2]) * depth / K[1, 1]
    z = depth

    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)   # (H*W, 3)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T + t[:, None]).T               # (H*W, 3)
    return pts_world


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Depth map → uint8 BGR heatmap (JET colormap, NaN-safe)."""
    d = depth.copy().astype(np.float32)
    valid = d > 0
    if valid.any():
        d_min, d_max = d[valid].min(), d[valid].max()
        d = np.clip((d - d_min) / (d_max - d_min + 1e-8), 0, 1)
    heatmap = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap[~valid] = 0
    return heatmap                                           # BGR, uint8

def save_one_image_canvas(
    output_dir: str,
    left_image:  np.ndarray,   # rendered depth (float [0,1] or uint8)
    tag: str,
    label: str = "DA3 Depth",
):
    """
    Save a single image with label.
    """
    img = left_image.copy()

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    img = img.astype(np.float32)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] == 3:
        img = img[..., ::-1]

    out_path = f"{output_dir}/render_compared/{tag}.png"
    imageio.imwrite(out_path, (img * 255).clip(0, 255).astype(np.uint8))
    print(f"  ✅  image saved → {out_path}") 

def save_outputs_canvas(
    output_dir: str,
    left_image:  np.ndarray,   # rendered depth (float [0,1] or uint8)
    right_image: np.ndarray,   # original RGB  (float [0,1] or uint8)
    tag: str,
    label: str = "DA3 Depth",
):
    """
    Save a side-by-side comparison PNG.
    Both images are converted to float32 [0,1] before concatenation.
    """
    left  = left_image.copy()
    right = right_image.copy()

    # ── normalise dtypes to float32 [0,1] ────────────────────────────────────
    if left.dtype == np.uint8:
        left = left.astype(np.float32) / 255.0
    if right.dtype == np.uint8:
        right = right.astype(np.float32) / 255.0

    # ── ensure both are float32 ───────────────────────────────────────────────
    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    # ── match spatial resolution (downsample left if 2× larger) ──────────────
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]

    if lh == 2 * rh:
        left = cv2.resize(left, (lw // 2, lh // 2))
    elif lh != rh:
        right = cv2.resize(right, (rw, lh))

    # ── add label to left panel ───────────────────────────────────────────────
    left_u8 = (left * 255).clip(0, 255).astype(np.uint8)
    cv2.putText(
        left_u8, label,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0,
        (255, 255, 255), 3, cv2.LINE_AA,
    )
    left = left_u8.astype(np.float32) / 255.0

    # ── ensure 3-channel RGB on both sides ───────────────────────────────────
    if left.ndim == 2:
        left = np.stack([left] * 3, axis=-1)
    if right.ndim == 2:
        right = np.stack([right] * 3, axis=-1)

    # ── handle BGR vs RGB (cv2 reads BGR, imageio writes RGB) ─────────────────
    # left came from cv2.applyColorMap → BGR, convert to RGB
    if left.shape[-1] == 3:
        left = left[..., ::-1]

    canvas = np.concatenate([left, right], axis=1)          # side-by-side
    out_path = f"{output_dir}/render_compared/{tag}.png"
    imageio.imwrite(out_path, (canvas * 255).clip(0, 255).astype(np.uint8))
    print(f"  ✅  canvas saved → {out_path}")


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular error between two rotation matrices (degrees)."""
    R_rel  = R1 @ R2.T
    trace  = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(trace)))


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """L2 distance between two translation vectors."""
    return float(np.linalg.norm(t1 - t2))


# =============================================================================
# STEP 1 — Load NPZ
# =============================================================================

print("\n" + "="*60)
print("STEP 1 — Load NPZ")
print("="*60)

data        = np.load(NPZ_PATH,mmap_mode='r')
depth_all   = data["depth"]          # (N, H, W)
print(data.keys())
#conf_all    = data["conf"]           # (N, H, W)
extr_da3    = extrinsics     # (N, 4, 4)  world-to-camera
intr_da3    = intrinsics    # (N, 3, 3)

N, H, W = depth_all.shape

print(f"  depth shape   : {depth_all.shape}  dtype={depth_all.dtype}")
#print(f"  conf  shape   : {conf_all.shape}")
print(f"  extrinsics    : {extr_da3.shape}")
print(f"  intrinsics    : {intr_da3.shape}")
print(f"  depth range   : [{depth_all[depth_all>0].min():.3f}, {depth_all.max():.3f}] m")
#print(f"  conf  range   : [{conf_all.min():.3f}, {conf_all.max():.3f}]")

# Check for NaN / Inf
nan_d = np.isnan(depth_all).sum()
inf_d = np.isinf(depth_all).sum()
print(f"  NaN in depth  : {nan_d}    Inf: {inf_d}")
if nan_d or inf_d:
    print("  ⚠️  Replacing NaN/Inf with 0")
    depth_all = np.nan_to_num(depth_all, nan=0.0, posinf=0.0, neginf=0.0)

print("---------------------------------------------")
print(f"After depth map resizing")


# Upsample each depth map to original resolution
original_height = 1080
original_width = 2048
upsampled_depths = []
for i, depth in enumerate(depth_all):
    # Calculate scale factors
    scale_h = original_height / depth.shape[0]
    scale_w = original_width / depth.shape[1]
    
    # Method 1: Using cv2 (faster)
    upsampled = cv2.resize(
        depth, 
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    upsampled_depths.append(upsampled)

# Save upsampled depths
upsampled_depths = np.array(upsampled_depths)
N, H, W = upsampled_depths.shape

print(f"  depth shape   : {upsampled_depths.shape}  dtype={upsampled_depths.dtype}")
print(f"  depth range   : [{upsampled_depths[upsampled_depths>0].min():.3f}, {upsampled_depths.max():.3f}] m")

# Save image witout resize and with resize
save_one_image_canvas(
    OUTPUT_DIR,
    left_image  = colorize_depth(depth_all[0]),
    tag         = f"depth_no_resize",
    label       = f"DA3 depth (no resize)",
)

save_one_image_canvas(
    OUTPUT_DIR,
    left_image  = colorize_depth(upsampled_depths[0]),
    tag         = f"depth_upsampled",
    label       = f"DA3 depth (upsampled)",
)

# =============================================================================
# STEP 2 — Load COLMAP ground truth
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Load COLMAP sparse reconstruction")
print("="*60)

try:
    from gsplat.examples.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary

    
    print(f"  COLMAP cameras : {len(colmap_cameras)}")
    print(f"  COLMAP images  : {len(colmap_images)}")
    print(f"  COLMAP points  : {len(colmap_points)}")
    COLMAP_AVAILABLE = True
    if COLMAP_AVAILABLE and len(colmap_images) > 0:
        # COLMAP stores images in a dict → sort by name for consistency
        first_im = sorted(colmap_images.values(), key=lambda im: im.name)[0]

        img_path = os.path.join(IMAGES_DIR, first_im.name)
        print(f"  First COLMAP image path: {img_path}")

        if os.path.exists(img_path):
            rgb = np.array(Image.open(img_path).convert("RGB"))
            print(f"  RGB shape: {rgb.shape}  dtype={rgb.dtype}")
        else:
            print("  ⚠️ Image file not found on disk.")

except Exception as e:
    print(f"  ⚠️  Could not load COLMAP data: {e}")
    COLMAP_AVAILABLE = False


# =============================================================================
# STEP 3 — Find RGB images on disk and match to NPZ indices
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Discover RGB images")
print("="*60)

exts  = ("*.png", "*.jpg")
img_paths = []
for ext in exts:
    img_paths += glob.glob(os.path.join(IMAGES_DIR, ext))
img_paths = sorted(img_paths)

print(f"  Found {len(img_paths)} images in {IMAGES_DIR}")
print(f"  NPZ contains   {N} depth maps")

if len(img_paths) == 0:
    print("  ⚠️  No images found — canvas comparison will be skipped.")

# Map: npz index → image path  (by sorted order)
def npz_idx_to_img_path(idx: int) -> str | None:
    if idx < len(img_paths):
        return img_paths[idx]
    return None


# =============================================================================
# STEP 4 — Per-image validation
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Per-image depth + pose validation")
print("="*60)

# Collect COLMAP poses in sorted-image order for comparison
colmap_w2c_list = []
colmap_K_list   = []
if COLMAP_AVAILABLE:
    sorted_colmap_imgs = sorted(colmap_images.values(), key=lambda im: im.name)
    for im in sorted_colmap_imgs:
        # Build world-to-camera extrinsic 4×4
        R_c = im.qvec2rotmat()
        t_c = im.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R_c
        w2c[:3, 3]  = t_c
        colmap_w2c_list.append(w2c)

        # Build intrinsic K from COLMAP camera model
        cam = colmap_cameras[im.camera_id]
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            K_c = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        elif cam.model in ("PINHOLE", "OPENCV", "RADIAL"):
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            K_c = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        else:
            K_c = None
        colmap_K_list.append(K_c)

MAX_COMPARE = min(N, len(colmap_w2c_list)) if COLMAP_AVAILABLE else 0
rot_errors   = []
trans_errors = []
fx_errors    = []

for i in range(min(N, 10)):          # print first 10
    d_i    = depth_all[i]
    # c_i    = conf_all[i]
    K_da3  = intr_da3[i]
    w2c_da3 = extr_da3[i]

    # ── depth stats ──────────────────────────────────────────────────────────
    valid_px = d_i[d_i > 0]
    pct_valid = 100 * len(valid_px) / d_i.size

    print(f"\n  [img {i:03d}]  valid={pct_valid:.1f}%  "
          f"depth=[{valid_px.min() if len(valid_px) else 0:.2f}, "
          f"{valid_px.max() if len(valid_px) else 0:.2f}]m  ")
          # f"conf_mean={c_i.mean():.3f}")

    # ── intrinsic check ───────────────────────────────────────────────────────
    print(f"           DA3 K  fx={K_da3[0,0]:.1f}  fy={K_da3[1,1]:.1f}  "
          f"cx={K_da3[0,2]:.1f}  cy={K_da3[1,2]:.1f}")

    # ── extrinsic check ───────────────────────────────────────────────────────
    # Verify w2c is a valid rigid transform: R should be orthogonal
    R_da3 = w2c_da3[:3, :3]
    orth_err = np.linalg.norm(R_da3 @ R_da3.T - np.eye(3))
    print(f"           R orthogonality error : {orth_err:.2e}  (< 1e-5 = good)")

    # ── COLMAP comparison ─────────────────────────────────────────────────────
    if COLMAP_AVAILABLE and i < len(colmap_w2c_list):
        w2c_col = colmap_w2c_list[i]
        R_col   = w2c_col[:3, :3]
        t_col   = w2c_col[:3, 3]
        t_da3   = w2c_da3[:3, 3]

        r_err = rotation_error_deg(R_da3, R_col)
        t_err = translation_error(t_da3, t_col)
        rot_errors.append(r_err)
        trans_errors.append(t_err)
        print(f"           COLMAP Δrot={r_err:.2f}°  Δtrans={t_err:.4f}m")

        if colmap_K_list[i] is not None:
            K_col  = colmap_K_list[i]
            fx_err = abs(K_da3[0,0] - K_col[0,0])
            fx_errors.append(fx_err)
            print(f"           COLMAP K  fx={K_col[0,0]:.1f}  fy={K_col[1,1]:.1f}  "
                  f"cx={K_col[0,2]:.1f}  cy={K_col[1,2]:.1f}   Δfx={fx_err:.2f}px")

if COLMAP_AVAILABLE and rot_errors:
    print(f"\n  ── Summary over first {len(rot_errors)} images ──")
    print(f"  Mean rotation error    : {np.mean(rot_errors):.3f}°  (< 1° = well-aligned)")
    print(f"  Mean translation error : {np.mean(trans_errors):.4f}m")
    if fx_errors:
        print(f"  Mean focal length error: {np.mean(fx_errors):.2f}px")


# =============================================================================
# STEP 5 — Visual canvas comparison (depth heatmap | RGB)
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Save left-right canvas comparisons")
print("="*60)

NUM_SAVE = min(N, 10)   # save first 10

for i in range(NUM_SAVE):
    d_i  = depth_all[i]
    #c_i  = conf_all[i]

    # ── left panel: depth heatmap (BGR uint8) ─────────────────────────────────
    depth_vis = colorize_depth(d_i)              # BGR uint8

    # ── right panel: RGB image ────────────────────────────────────────────────
    img_path = npz_idx_to_img_path(i)
    if img_path is not None:
        rgb = np.array(Image.open(img_path).convert("RGB"))  # uint8 RGB
    else:
        # Fallback: grey placeholder
        rgb = np.ones((H, W, 3), dtype=np.uint8) * 128

    # ── confidence overlay (optional — fades out low-conf pixels) ────────────
    # conf_norm = (c_i / c_i.max() * 255).astype(np.uint8) if c_i.max() > 0 else np.zeros_like(c_i, dtype=np.uint8)
    # conf_rgb  = cv2.cvtColor(conf_norm, cv2.COLOR_GRAY2BGR)   # for reference
    upsampled_depth = cv2.resize(
        depth_vis, 
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    upsampled_rgb = cv2.resize(
        rgb, 
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    print(f"DEPTH SHAPE {upsampled_depth.shape}")
    print(f"RGB SHAPE {rgb.shape}")
    # Main canvas: depth | rgb
    save_outputs_canvas(
        OUTPUT_DIR,
        left_image  = upsampled_depth,     # BGR uint8 → function converts to RGB float
        right_image = upsampled_rgb,           # RGB uint8
        tag         = f"depth_vs_rgb_{i:03d}",
        label       = f"DA3 depth [{i}]",
    )

    # Bonus canvas: confidence | rgb
    # save_outputs_canvas(
    #     OUTPUT_DIR,
    #     left_image  = conf_rgb,
    #     right_image = rgb,
    #     tag         = f"conf_vs_rgb_{i:03d}",
    #     label       = f"DA3 conf [{i}]",
    # )


# =============================================================================
# STEP 6 — Back-project depth to 3-D and save point cloud
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Back-project and save point clouds")
print("="*60)

NUM_PLY = min(N, 3)    # save first 3 as .ply

for i in range(NUM_PLY):
    d_i   = depth_all[i]
    K_i   = intr_da3[i]
    w2c_i = extr_da3[i]

    # DA3 extrinsics = world-to-camera (w2c).
    # depth_to_3d needs camera-to-world (c2w) = inv(w2c)
    c2w_i = np.linalg.inv(w2c_i)

    pts = depth_to_3d(d_i, K_i, c2w_i)

    # Filter: keep only points with valid depth
    valid_mask = d_i.reshape(-1) > 0
    pts = pts[valid_mask]

    # Colour from RGB image
    img_path = npz_idx_to_img_path(i)
    if img_path is not None:
        rgb_full = np.array(Image.open(img_path).convert("RGB").resize((W, H)))
        colors   = rgb_full.reshape(-1, 3)[valid_mask].astype(np.float64) / 255.0
    else:
        colors = np.ones((pts.shape[0], 3)) * 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ply_path = f"{OUTPUT_DIR}/pointclouds/scene1_img{i:03d}.ply"
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"  ✅  {pts.shape[0]:,} points → {ply_path}")


# =============================================================================
# STEP 7 — Quick sanity: re-project a 3D point back to 2D
# =============================================================================

print("\n" + "="*60)
print("STEP 7 — Re-projection sanity check")
print("="*60)

i = 0
K_i   = intr_da3[i]
w2c_i = extr_da3[i]
d_i   = depth_all[i]

# Pick the pixel with median depth as reference
valid_yx  = np.argwhere(d_i > 0)
if len(valid_yx) > 0:
    mid       = valid_yx[len(valid_yx) // 2]
    py, px    = int(mid[0]), int(mid[1])
    d_ref     = float(d_i[py, px])

    # Back-project to world
    c2w_i = np.linalg.inv(w2c_i)
    Xc = (px - K_i[0,2]) * d_ref / K_i[0,0]
    Yc = (py - K_i[1,2]) * d_ref / K_i[1,1]
    Zc = d_ref
    pt_cam   = np.array([Xc, Yc, Zc])
    pt_world = c2w_i[:3,:3] @ pt_cam + c2w_i[:3,3]

    # Re-project back with w2c
    pt_cam2  = w2c_i[:3,:3] @ pt_world + w2c_i[:3,3]
    u = K_i[0,0] * pt_cam2[0] / pt_cam2[2] + K_i[0,2]
    v = K_i[1,1] * pt_cam2[1] / pt_cam2[2] + K_i[1,2]

    err = np.sqrt((u - px)**2 + (v - py)**2)
    print(f"  Reference pixel : ({px}, {py})  depth={d_ref:.3f}m")
    print(f"  Re-projected    : ({u:.2f}, {v:.2f})")
    print(f"  Re-proj error   : {err:.4f}px  (< 0.01px = perfect)")
else:
    print("  ⚠️  No valid depth pixels found in image 0.")


# =============================================================================
# DONE
# =============================================================================

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print(f"  outputs → {OUTPUT_DIR}/")
print("="*60 + "\n")