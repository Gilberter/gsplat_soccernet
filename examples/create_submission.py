
import os
import zipfile
import argparse
from pathlib import Path


def find_render_dirs(scene_path, keyword=None, parent_filter=None):
    """
    Find all 'renders_ours' folders.
    Optionally filter by keyword in full path and parent folder name.
    """
    candidates = []

    for root, dirs, files in os.walk(scene_path):
        root_path = Path(root)

        # ✅ Only look for renders_ours
        if root_path.name != "renders_ours":
            continue

        # Optional: filter full path (e.g. results-depth-prior)
        if keyword and keyword not in str(root_path):
            continue

        # Optional: filter parent folder (e.g. antialiased_appopt_s35000)
        if parent_filter and root_path.parent.name != parent_filter:
            continue

        candidates.append(root_path)

    return candidates


def create_submission_zip(base_dir, scenes, results_folder=None, get_renders_dir=None):
    output_folder = os.path.join(base_dir, "submissions")
    os.makedirs(output_folder, exist_ok=True)

    zip_path = os.path.join(output_folder, "submission.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:

        for scene in scenes:
            scene_path = os.path.join(base_dir, scene)

            if not os.path.exists(scene_path):
                print(f"Skipping {scene}, not found")
                continue

            # 🔍 Find candidate render folders
            render_dirs = find_render_dirs(
                scene_path,
                keyword=results_folder,
                parent_filter=get_renders_dir
            )

            if not render_dirs:
                print(f"No renders found in {scene}")
                continue

            for renders_dir in render_dirs:
                pngs = sorted(renders_dir.glob("*.png"))

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
    )

    parser.add_argument(
        "--results_folder",
        type=str,
        default=None,
        help="Optional keyword filter (e.g. results-depth-prior)"
    )

    parser.add_argument(
        "--get_renders_dir",
        type=str,
        default=None,
        help="Exact folder name to match (e.g. antialiased_appopt_s35000)"
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
    )

    args = parser.parse_args()

    create_submission_zip(
        args.base_dir,
        args.scenes,
        args.results_folder,
        args.get_renders_dir
    )


if __name__ == "__main__":
    main()