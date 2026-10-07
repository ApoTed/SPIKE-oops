"""
make_oops_population.py
=======================
Builds the Oops! evaluation population we use, from the official Oops!
annotation files alone.

Rule: a video is included if it is in the dataset's own filtered validation
list (val_filtered.txt) AND has at least one valid annotator label (a label of
-1 means "the annotator found no failure"). On the released annotations this
gives 4,020 videos.

Why this population: the official Oops! localization protocol (Epstein et al.,
CVPR 2020, Sec. 5.3) scores a prediction against the annotated transition
times, so it is only defined on clips that have at least one. The filtered list
additionally drops the clips the dataset authors flag as unreliable (failure at
the very start or end of the clip, i.e. scene-detection errors).

Output format, as expected by build_spike_rl_input.py:
  {"videos": [{"basename": ..., "split": "val", "t_med": ..., "t": [...]}, ...]}

Usage:
  python3 make_oops_population.py \
      --annotations /path/to/oops_dataset/annotations \
      --out oops_val_population.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--annotations", type=Path, required=True,
                    help="Oops! annotations folder, containing val_filtered.txt "
                         "and transition_times.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="where to write the population JSON")
    args = ap.parse_args()

    names = [l.strip() for l in open(args.annotations / "val_filtered.txt",
                                     encoding="utf-8") if l.strip()]
    tt = json.load(open(args.annotations / "transition_times.json", encoding="utf-8"))

    videos, n_no_entry, n_no_label = [], 0, 0
    for b in names:
        ann = tt.get(b)
        if ann is None:
            n_no_entry += 1
            continue
        labels = [t for t in (ann.get("t") or []) if t is not None and t != -1]
        if not labels:
            n_no_label += 1
            continue
        videos.append({"basename": b, "split": "val",
                       "t_med": float(median(labels)), "t": ann.get("t")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"rule": "val_filtered.txt AND >=1 valid label",
                        "n_videos": len(videos)},
               "videos": videos},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"filtered validation list: {len(names)} videos")
    print(f"  without an annotation entry: {n_no_entry}")
    print(f"  with no valid label (all -1): {n_no_label}")
    print(f"population written to {args.out}: {len(videos)} videos")


if __name__ == "__main__":
    main()
