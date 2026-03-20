
import os
import re
import shutil
from collections import defaultdict


IMAGE_DIR   = "/disk/SN-NVS-2026-raw/scene-1/images_4"
TXT_FILE    = "/home/hensemberk/dev/Soccernet/gsplat/examples/keypoint_matches.txt"
OUTPUT_BASE = "/disk/SN-NVS-2026-raw/results-soccernet/matches_images_"

TOP_N = 10


# ---------------------------------------------------------------------------
# Step 1: Parse the SUMMARY block — fastest way to rank pairs by match count
# ---------------------------------------------------------------------------

def parse_summary(txt_file):
    """
    Read the SUMMARY block at the bottom of the log file.
    Returns a list of (name_i, name_j, match_count) sorted best-first.

    Summary lines look like:
      image_001.jpg              <-> image_007.jpg               1234 matches
    """
    pairs = []
    in_summary = False

    # Regex: two filenames separated by <-> and an integer at the end
    line_re = re.compile(
        r"^\s*(\S+)\s+<->\s+(\S+)\s+(\d+)\s+matches"
    )

    with open(txt_file, "r") as f:
        for line in f:
            if "SUMMARY" in line:
                in_summary = True
                continue
            if not in_summary:
                continue
            m = line_re.match(line)
            if m:
                name_i      = m.group(1)
                name_j      = m.group(2)
                match_count = int(m.group(3))
                pairs.append((name_i, name_j, match_count))

    # Sort descending by match count
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Fallback: if no SUMMARY block, count matches from the data lines directly
# ---------------------------------------------------------------------------

def parse_data_lines(txt_file):
    """
    Count matches per image pair from the raw data lines.
    Used as fallback if the SUMMARY block is absent.

    Data line format:
      pt3d=  142 | img_i.jpg   uv=( ...) | img_j.jpg   uv=( ...) | xyz=(...) err=...
    """
    counts = defaultdict(int)

    # Match: pt3d=NNN | NAME1 uv=... | NAME2 uv=...
    line_re = re.compile(
        r"pt3d=\s*\d+\s*\|\s*(\S+)\s+uv=.*?\|\s*(\S+)\s+uv="
    )

    with open(txt_file, "r") as f:
        for line in f:
            m = line_re.search(line)
            if m:
                name_i = m.group(1)
                name_j = m.group(2)
                key    = (name_i, name_j) if name_i <= name_j else (name_j, name_i)
                counts[key] += 1

    pairs = [(k[0], k[1], v) for k, v in counts.items()]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Step 2: Parse all match detail lines for a specific set of pairs
# ---------------------------------------------------------------------------

def parse_match_details(txt_file, target_pairs):
    """
    Do a single pass through the data lines and collect every match entry
    (pt3d, uv_i, uv_j, xyz, err) for each pair in target_pairs.

    Args:
        txt_file:     path to keypoint_matches.txt
        target_pairs: set of (name_i, name_j) tuples — canonical sorted order

    Returns:
        details: dict {(name_i, name_j) -> list of dicts}
            Each dict:
                "pt3d_id": int
                "uv_i":    (float, float)
                "uv_j":    (float, float)
                "xyz":     (float, float, float)
                "error":   float
    """
    details = defaultdict(list)

    # pt3d=    142 | img_i.jpg   uv=( 412.30, 280.10) | img_j.jpg   uv=( 388.70, 291.40) | xyz=(1.234,-0.567,3.210) err=0.412
    line_re = re.compile(
        r"pt3d=\s*(\d+)\s*\|"
        r"\s*(\S+)\s+uv=\(\s*([-\d.]+),\s*([-\d.]+)\)\s*\|"
        r"\s*(\S+)\s+uv=\(\s*([-\d.]+),\s*([-\d.]+)\)\s*\|"
        r"\s*xyz=\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)"
        r"\s*err=([-\d.]+)"
    )

    with open(txt_file, "r") as f:
        for line in f:
            if not line.startswith("pt3d="):
                continue
            m = line_re.search(line)
            if not m:
                continue

            pt3d_id = int(m.group(1))
            name_a  = m.group(2)
            uv_a    = (float(m.group(3)), float(m.group(4)))
            name_b  = m.group(5)
            uv_b    = (float(m.group(6)), float(m.group(7)))
            xyz     = (float(m.group(8)), float(m.group(9)), float(m.group(10)))
            error   = float(m.group(11))

            # Canonical ordering
            if name_a <= name_b:
                key, uv_i, uv_j = (name_a, name_b), uv_a, uv_b
            else:
                key, uv_i, uv_j = (name_b, name_a), uv_b, uv_a

            if key in target_pairs:
                details[key].append({
                    "pt3d_id": pt3d_id,
                    "uv_i":    uv_i,
                    "uv_j":    uv_j,
                    "xyz":     xyz,
                    "error":   error,
                })

    return details


# ---------------------------------------------------------------------------
# Step 3: Resolve image filenames to actual paths in IMAGE_DIR
# ---------------------------------------------------------------------------

def find_image(image_dir, name):
    """
    Given a filename like '000000.png' or 'frame_001.jpg', find it under
    image_dir. Handles cases where the name has no extension by trying
    common extensions.
    """
    # Direct match
    candidate = os.path.join(image_dir, name)
    if os.path.exists(candidate):
        return candidate

    # Strip extension and try common image types
    stem = os.path.splitext(name)[0]
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Step 3: Copy images into output subfolders
# ---------------------------------------------------------------------------

def copy_best_pairs(txt_file, image_dir, output_base, top_n=10):
    """
    Main function: parse txt, pick top N pairs, copy images.

    Output structure:
        output_base_01/   ← rank 1 pair
            image_A.png
            image_B.png
            info.txt      ← match count + uv info
        output_base_02/
            ...
    """
    print(f"Reading: {txt_file}")

    # Try SUMMARY block first (fast), fall back to full parse
    pairs = parse_summary(txt_file)
    if not pairs:
        print("  No SUMMARY block found — counting from data lines (slower)...")
        pairs = parse_data_lines(txt_file)

    if not pairs:
        print("ERROR: No matched pairs found in file.")
        return

    print(f"  Found {len(pairs)} view pairs total")
    print(f"  Selecting top {top_n}\n")

    selected     = pairs[:top_n]
    target_pairs = {(ni, nj) for ni, nj, _ in selected}

    # Single pass through the file to collect all match details for selected pairs
    print("  Parsing match details (pt3d / uv / xyz) ...")
    details = parse_match_details(txt_file, target_pairs)
    print(f"  Done.\n")

    for rank, (name_i, name_j, count) in enumerate(selected, start=1):

        out_dir = f"{output_base}{rank:02d}"
        os.makedirs(out_dir, exist_ok=True)

        path_i = find_image(image_dir, name_i)
        path_j = find_image(image_dir, name_j)

        status_i = "OK" if path_i else "NOT FOUND"
        status_j = "OK" if path_j else "NOT FOUND"

        if path_i:
            shutil.copy2(path_i, os.path.join(out_dir, name_i))
        if path_j:
            shutil.copy2(path_j, os.path.join(out_dir, name_j))

        pair_matches = details.get((name_i, name_j), [])

        # Write info.txt with full match details
        info_path = os.path.join(out_dir, "info.txt")
        with open(info_path, "w") as f:
            # Header
            f.write(f"{'='*70}\n")
            f.write(f"Rank:          {rank}\n")
            f.write(f"Match count:   {count}\n")
            f.write(f"Image i:       {name_i}  [{status_i}]\n")
            f.write(f"Image j:       {name_j}  [{status_j}]\n")
            f.write(f"Source dir:    {image_dir}\n")
            f.write(f"Output dir:    {out_dir}\n")
            f.write(f"{'='*70}\n\n")

            # Column header
            f.write(
                f"{'#':>5}  "
                f"{'pt3d_id':>9}  "
                f"{'uv_i_x':>10}  {'uv_i_y':>10}  "
                f"{'uv_j_x':>10}  {'uv_j_y':>10}  "
                f"{'xyz_x':>10}  {'xyz_y':>10}  {'xyz_z':>10}  "
                f"{'err':>8}\n"
            )
            f.write(f"{'-'*100}\n")

            # One row per matched 3D point
            for idx, match in enumerate(pair_matches, start=1):
                f.write(
                    f"{idx:>5}  "
                    f"{match['pt3d_id']:>9}  "
                    f"{match['uv_i'][0]:>10.2f}  {match['uv_i'][1]:>10.2f}  "
                    f"{match['uv_j'][0]:>10.2f}  {match['uv_j'][1]:>10.2f}  "
                    f"{match['xyz'][0]:>10.3f}  {match['xyz'][1]:>10.3f}  {match['xyz'][2]:>10.3f}  "
                    f"{match['error']:>8.4f}\n"
                )

            f.write(f"\nTotal: {len(pair_matches)} matches listed\n")

        print(f"  [{rank:02d}] {count:6d} matches | {name_i} [{status_i}]"
              f"  +  {name_j} [{status_j}]  ->  {out_dir}")

    print(f"\nDone. Results in: {output_base}01 ... {output_base}{top_n:02d}")



import os
import time
import numpy as np
from collections import defaultdict
from itertools import combinations


OUTPUT_DIR = "/home/hensemberk/dev/Soccernet/gsplat/examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE   = os.path.join(OUTPUT_DIR, "keypoint_matches.txt")


# ---------------------------------------------------------------------------
# Fast builder — vectorized, no Python inner loop over pairs
# ---------------------------------------------------------------------------

def build_keypoint_pairs_fast(images, points3D, log_every=50):
    """
    Vectorized version of build_keypoint_pairs.

    Key speedup: instead of a Python double for-loop over track pairs,
    we use itertools.combinations on the (usually small) track array,
    which is much faster than nested Python loops, and we batch-write
    to the log file every `log_every` points instead of per-pair.

    Args:
        images:     dict {image_id -> Image namedtuple}
        points3D:   dict {point3D_id -> Point3D namedtuple}
        log_every:  int, print + write to file every N points processed

    Returns:
        keypoint_pairs: dict {(name_i, name_j) -> list of dicts}
    """
    id_to_name     = {img.id: img.name for img in images.values()}
    # Pre-cache xys arrays for fast lookup
    id_to_xys      = {img.id: img.xys for img in images.values()}

    keypoint_pairs = defaultdict(list)
    point_ids      = list(points3D.keys())
    total          = len(point_ids)

    t0             = time.time()
    batch_lines    = []   # buffer log lines, flush every log_every points

    with open(LOG_FILE, "w") as log:
        log.write(f"Colmap keypoint matches\n")
        log.write(f"Total 3D points: {total}\n")
        log.write(f"{'='*70}\n\n")

        for i, pt3d_id in enumerate(point_ids):
            point         = points3D[pt3d_id]
            track_img_ids = point.image_ids       # np.array [track_length]
            track_pt2_idx = point.point2D_idxs    # np.array [track_length]
            track_length  = len(track_img_ids)

            if track_length < 2:
                # Progress tick even for skipped points
                if (i + 1) % log_every == 0:
                    _flush(log, batch_lines, i, total, t0)
                    batch_lines = []
                continue

            # --- All pairs within the track using combinations ---
            # combinations gives indices (a, b) with a < b, no Python double loop
            indices = np.array(list(combinations(range(track_length), 2)))
            # indices shape: [num_pairs, 2]

            a_idx = indices[:, 0]
            b_idx = indices[:, 1]

            img_ids_a  = track_img_ids[a_idx]   # [num_pairs]
            img_ids_b  = track_img_ids[b_idx]   # [num_pairs]
            pt2d_ids_a = track_pt2_idx[a_idx]   # [num_pairs]
            pt2d_ids_b = track_pt2_idx[b_idx]   # [num_pairs]

            for k in range(len(indices)):
                id_i = img_ids_a[k]
                id_j = img_ids_b[k]

                if id_i not in id_to_xys or id_j not in id_to_xys:
                    continue

                uv_i   = tuple(id_to_xys[id_i][pt2d_ids_a[k]])
                uv_j   = tuple(id_to_xys[id_j][pt2d_ids_b[k]])
                name_i = id_to_name[id_i]
                name_j = id_to_name[id_j]

                if name_i <= name_j:
                    key, uv_first, uv_second = (name_i, name_j), uv_i, uv_j
                else:
                    key, uv_first, uv_second = (name_j, name_i), uv_j, uv_i

                entry = {
                    "uv_i":       uv_first,
                    "uv_j":       uv_second,
                    "point3D_id": pt3d_id,
                    "xyz":        point.xyz,
                    "error":      float(point.error),
                }
                keypoint_pairs[key].append(entry)

                # Buffer log line for this match
                batch_lines.append(
                    f"pt3d={pt3d_id:7d} | "
                    f"{key[0]:25s} uv=({uv_first[0]:8.2f},{uv_first[1]:8.2f}) | "
                    f"{key[1]:25s} uv=({uv_second[0]:8.2f},{uv_second[1]:8.2f}) | "
                    f"xyz=({point.xyz[0]:.3f},{point.xyz[1]:.3f},{point.xyz[2]:.3f}) "
                    f"err={float(point.error):.3f}\n"
                )

            # --- Every log_every points: print progress + flush to file ---
            if (i + 1) % log_every == 0:
                _flush(log, batch_lines, i, total, t0)
                batch_lines = []

        # Final flush for remaining lines
        if batch_lines:
            _flush(log, batch_lines, total - 1, total, t0)

        # Write summary footer
        summary = _build_summary(keypoint_pairs, time.time() - t0)
        log.write(summary)
        print(summary)

    return dict(keypoint_pairs)


def _flush(log, batch_lines, i, total, t0):
    """Print progress to stdout and write buffered lines to log file."""
    elapsed  = time.time() - t0
    rate     = (i + 1) / elapsed if elapsed > 0 else 0
    eta      = (total - i - 1) / rate if rate > 0 else 0
    progress = f"[{i+1:6d}/{total}]  {elapsed:6.1f}s elapsed  " \
               f"{rate:6.1f} pts/s  ETA {eta:5.1f}s"
    print(progress)
    log.write(progress + "\n")
    log.writelines(batch_lines)
    log.flush()   # force write to disk immediately


def _build_summary(keypoint_pairs, elapsed):
    lines = [f"\n{'='*70}\n", f"  SUMMARY\n", f"{'='*70}\n"]
    total_pairs = 0
    for (name_i, name_j), pairs in sorted(keypoint_pairs.items()):
        lines.append(
            f"  {name_i:30s} <-> {name_j:30s}  {len(pairs):7d} matches\n"
        )
        total_pairs += len(pairs)
    lines.append(f"{'='*70}\n")
    lines.append(f"  View pairs:          {len(keypoint_pairs)}\n")
    lines.append(f"  Total match pairs:   {total_pairs}\n")
    lines.append(f"  Total time:          {elapsed:.2f}s\n")
    lines.append(f"{'='*70}\n")
    return "".join(lines)

print(f"{len(points)} points")
print(f"Logging to: {LOG_FILE}\n")

t_start        = time.time()
keypoint_pairs = build_keypoint_pairs_fast(images, points, log_every=50)
t_total        = time.time() - t_start

print(f"\nDone in {t_total:.2f}s")
print(f"Log saved to: {LOG_FILE}")

# Inspect first pair
first_key   = list(keypoint_pairs.keys())[0]
first_pairs = keypoint_pairs[first_key]
m           = first_pairs[0]
print(f"\nExample pair: {first_key[0]}  <->  {first_key[1]}")
print(f"  Total matches:      {len(first_pairs)}")
print(f"  First match uv_i:   {m['uv_i']}")
print(f"  First match uv_j:   {m['uv_j']}")
print(f"  3D xyz:             {m['xyz']}")
print(f"  reprojection err:   {m['error']:.4f} px")



image_path_scene = "/disk/SN-NVS-2026-raw/scene-1/images_4"
text1_points_images_match = "/home/hensemberk/dev/Soccernet/gsplat/examples/keypoint_matches.txt"

out_put_images_folder = "/disk/SN-NVS-2026-raw/results-soccernet/matches_images_"


copy_best_pairs(
        txt_file    = TXT_FILE,
        image_dir   = IMAGE_DIR,
        output_base = OUTPUT_BASE,
        top_n       = TOP_N,
    )