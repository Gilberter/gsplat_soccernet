import os
import zipfile
import argparse
from pathlib import Path


def create_submission_zip(base_dir, scenes, results_folder, get_renders_dir):
    output_folder = os.path.join(f"{base_dir}/submissions",results_folder)
    os.makedirs(output_folder, exist_ok=True)
    zip_path = os.path.join(output_folder, "submission.zip")
    
    
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:

        for scene in scenes:

            renders_dir = os.path.join(base_dir, scene, results_folder, get_renders_dir)

            if not os.path.exists(renders_dir):
                print(f"Skipping {scene}, {renders_dir} not found")
                continue

            pngs = sorted(Path(renders_dir).glob("*.png"))

            for png in pngs:
                arcname = os.path.join(scene, "renders", png.name)
                zipf.write(png, arcname)
                print(f"Added {arcname}")

    print(f"\nZip created: {zip_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/disk/SN-NVS-2026-raw",
        help="Base directory containing scene folders"
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        help="Results folder inside each scene (e.g. results_factor_1)"
    )

    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "scene-1-challenge",
            "scene-2-challenge",
            "scene-3-challenge",
            "scene-4-challenge",
            "scene-5-challenge"
        ],
        help="Scene names"
    )

    parser.add_argument(
        "--get_renders_dir",
        type=str,
        default="renders_ours",
        help="Scene names"
    )

    args = parser.parse_args()

    create_submission_zip(args.base_dir, args.scenes, args.results_folder,args.get_renders_dir)


if __name__ == "__main__":
    main()