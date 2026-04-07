
import os
from utils_eval_2dgs_challenge import (
    load_challenges,
    save_outputs,
    render_camera_2dgs,
    load_splats,
    save_outputs_canvas,
)
import cv2
import argparse
import torch
import imageio.v3 as iio
import numpy as np
import shutil

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.data_dir, exist_ok=True)

    # --- Load 2DGS Gaussians ---
    means, quats, scales, opacities, colors, sh_degree, app_module, features = load_splats(
        args.ckpt, device, model_type="2dgs"
    )

    if sh_degree is not None:
        sh_degree_render = min(args.max_sh_degree, sh_degree)
    else:
        sh_degree_render = None

    c2w_mats, ks_list, imsize_list, image_ids = load_challenges(
        f"{args.data_dir}/sparse/0", factor=2
    )
    n_cameras = len(image_ids)

    # specific indices passed via --indices
    if args.indices is not None and len(args.indices) == 2:
        start, end = args.indices
        indices = np.arange(start, end + 1)  # +1 to include the end value
    elif args.indices is not None:
        indices = np.array(args.indices)
    else:
        # default if not index pass
        indices = np.arange(n_cameras)

    # out of bounds index crash
    indices = indices[indices < n_cameras]

    print(f"Rendering {len(indices)} cameras from {args.data_dir}")

    path_renders = sorted(os.listdir(f"{args.data_dir}/renders"))

    for i, idx in enumerate(indices):
        c2w = c2w_mats[idx]  # (4, 4) float64
        K = ks_list[idx].copy()  # (3, 3) undistorted
        width, height = imsize_list[idx]

        # Diagnostic
        print(
            f"  cam {idx} | image_id={image_ids[idx]} | "
            f"{width}x{height} | t={c2w[:3,3].round(2)} k={K.shape}"
        )

        render_out, n = render_camera_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree_render,
            c2w,
            K,
            width,
            height,
            near=args.near,
            far=args.far,
            device=device,
            app_module=app_module,
            features=features,
        )

        print(f"  SHAPE render_out {render_out.shape}")
        print(f"  MAX render_out {render_out.max()}")

        if isinstance(render_out, torch.Tensor):
            print("  Converting from Tensor to numpy")
            render_out = render_out.detach().cpu().numpy()

        # Shape guard: must be (H, W, 3)
        if render_out.ndim == 4:
            print(f"  Removing batch dimension")
            render_out = render_out.squeeze(0)  # remove batch dim if present

        if render_out.shape[-1] > 3:
            render_out = render_out[..., :3]  # drop depth / alpha channel
        elif render_out.shape[-1] < 3:
            raise ValueError(
                f"Unexpected render output channels: {render_out.shape}. "
                "Expected (H, W, 3) or (H, W, 4)."
            )

        assert (
            render_out.ndim == 3 and render_out.shape[-1] == 3
        ), f"render_out has unexpected shape after fix: {render_out.shape}"

        left_image = render_out.copy()  # (H, W, 3)
        tag = f"{i:05d}"

        image_right = f"{args.data_dir}/renders/{path_renders[i]}"
        img_rendered_right = iio.imread(image_right)

        print(f"  SHAPE render_out {render_out.shape}")
        print(f"  SHAPE image_right {img_rendered_right.shape}")
        print(f"  SHAPE img_rendered_right MAX {img_rendered_right.max()}")
        print(f"  SHAPE render_out MAX {render_out.max()}")

        if args.specific:
            output_dir = args.result_folder
        else:
            output_dir = os.path.join(args.data_dir, args.result_folder)

        save_outputs(output_dir, tag, left_image)
        save_outputs_canvas(output_dir, left_image, img_rendered_right, tag)

        for ckpt_path in args.ckpt:
            filename = os.path.basename(ckpt_path)
            dst_path = os.path.join(output_dir, filename)

            if not os.path.exists(dst_path):
                shutil.copy2(ckpt_path, dst_path)

        print(
            f"  [{i+1}/{len(indices)}] | {width}x{height} | rendered {n} Gaussians"
        )

    print(
        f"\n✅ Done. Scene {args.data_dir} outputs in: {args.result_folder if args.result_folder != '' else 'Same input folder'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to checkpoint .pt files",
    )
    parser.add_argument("--max_sh_degree", type=int, default=3)

    # Scene / COLMAP
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Challenge Directory")
    parser.add_argument(
        "--factor", type=int, default=2, help="Downscale factor (must match training)"
    )

    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit camera indices (used when --split=custom)",
    )
    parser.add_argument("--result_folder", type=str, default="")

    # Render params
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--specific", action="store_true")

    args = parser.parse_args()
    main(args)