"""
build_spike_rl_input.py
=======================
Adapter: converts a population JSON (e.g. from make_oops_population.py) into
the input that SPIKE-RL's own inference.py expects, so the unmodified SPIKE-RL
code runs on exactly the videos we choose.

SPIKE-RL's Oops! loader (src/open_r1_video/inference/oops.py) expects:
  - json_path: a JSON list of dicts, each with "set_id", "index" and
    "transition" (the failure time in seconds);
  - video_root: a directory laid out as
    {video_root}/{set_id}_merged/{index}_E_merged.mp4
    (the folder convention of their preprocessed Oops! release, not the flat
    oops_video/{split}/{basename}.mp4 layout of the original dataset).

Rather than modifying their path-construction code, this script builds a small
symlink farm mapping the original video files into that layout, plus the
matching JSON. Only clips with an annotated failure are included: the loader
calls float(transition) unconditionally, and the official Oops! protocol only
scores failure clips anyway.

Usage:
  python3 build_spike_rl_input.py \
      --subset oops_val_population.json \
      --video-root /path/to/oops_dataset/oops_video \
      --out-json spike_rl_input/oops_val.json \
      --out-video-root spike_rl_input/oops_val_videos

Then run SPIKE-RL's inference.py on the generated paths (see RUNNING.md).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, required=True,
                    help="population JSON with a 'videos' list "
                         "(basename, split, t_med), e.g. from make_oops_population.py")
    ap.add_argument("--video-root", type=Path, required=True,
                    help="root of the original Oops! videos, laid out as "
                         "{video_root}/{split}/{basename}.mp4")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="where to write the JSON in SPIKE-RL's format")
    ap.add_argument("--out-video-root", type=Path, required=True,
                    help="where to build the symlink farm "
                         "{basename}_merged/0_E_merged.mp4")
    ap.add_argument("--reachable-only", action="store_true",
                    help="keep only videos with reachable=true, if the population "
                         "file has that field. Default: every failure clip.")
    args = ap.parse_args()

    with open(args.subset, "r", encoding="utf-8") as f:
        subset = json.load(f)
    videos = subset["videos"]

    args.out_video_root.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    n_skipped_no_failure = 0
    n_skipped_unreachable = 0
    n_skipped_missing = 0

    for v in videos:
        t_med = v.get("t_med")
        if t_med is None:
            n_skipped_no_failure += 1
            continue
        if args.reachable_only and not v.get("reachable", True):
            n_skipped_unreachable += 1
            continue

        basename = v["basename"]
        split = v.get("split", "val")
        src = args.video_root / split / f"{basename}.mp4"
        if not src.exists():
            print(f"  [MISS] video not found: {src}")
            n_skipped_missing += 1
            continue

        dest_dir = args.out_video_root / f"{basename}_merged"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "0_E_merged.mp4"
        if not dest.exists():
            try:
                os.symlink(src.resolve(), dest)
            except OSError as e:
                print(f"  [WARN] symlink failed for {basename} ({e}), copying instead")
                import shutil
                shutil.copy2(src, dest)

        entries.append({
            "set_id": basename,
            "index": "0",
            "transition": float(t_med),
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {len(entries)} entries to {args.out_json}")
    print(f"video farm in {args.out_video_root}")
    print(f"skipped: no-failure={n_skipped_no_failure}  "
          f"unreachable={n_skipped_unreachable}  missing={n_skipped_missing}")


if __name__ == "__main__":
    main()
