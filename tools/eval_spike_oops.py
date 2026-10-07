"""
eval_spike_oops.py
==================
Aggregates the SPIKE-RL output shards on Oops! and reports the results under
two ground-truth definitions, side by side:

  * the OFFICIAL Oops! protocol (Epstein et al., CVPR 2020, Sec. 5.3): a
    prediction is correct if it falls within 0.25 s / 1 s of ANY of the (up to
    three) annotator labels;
  * the metric SAVED BY THE RELEASED CODE: accuracy_at_delta_* computed against
    the code's own ground truth (the sampled frames falling inside the
    continuous window [0.8 * t, 0.8 * duration]), plus iou_peak and
    contiguous_iou.

The peak selection is identical in both cases (argmax of surprise_scores):
only the ground-truth definition changes, so the comparison is clean.

Usage:
  python3 eval_spike_oops.py \
      --oops-dir results/oops \
      --transition-times /path/to/oops_dataset/annotations/transition_times.json \
      [--shards valALL] [--population oops_val_population.json] [--dump per_video.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def basename_from_video_path(video_path: str) -> str:
    """.../{basename}_merged/0_E_merged.mp4 -> basename"""
    return re.sub(r"_merged$", "", Path(video_path).parent.name)


def official_correct(pred_time, raw_times, thr):
    valid = [t for t in raw_times if t is not None and t != -1]
    if pred_time is None or not valid:
        return None
    return min(abs(pred_time - t) for t in valid) <= thr


def load_shards(oops_dir: Path, pattern: str):
    """One record per video: the prediction plus the metrics saved by the code."""
    per_video, dupes = {}, 0
    shards = []
    for d in sorted(oops_dir.iterdir()):
        if not d.is_dir() or pattern not in d.name:
            continue
        res = d / "results.json"
        fin = d / "results_final.json"
        if not res.exists():
            continue
        custom = {}
        if fin.exists():
            try:
                custom = json.load(open(fin, encoding="utf-8"))
            except Exception:
                custom = {}
        n_shard = 0
        for line in open(res, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vp = r.get("video_path")
            scores = r.get("surprise_scores") or []
            idx = r.get("frame_indices") or []
            if not vp or not scores or not idx:
                continue
            base = basename_from_video_path(vp)
            fps = r.get("fps") or 30.0
            rec = {
                "basename": base,
                "shard": d.name,
                "fps": fps,
                "n_frames_scored": len(scores),
                "pred_time": idx[int(np.argmax(scores))] / fps,
                "custom": custom.get(vp, {}),
            }
            if base in per_video:
                dupes += 1
                continue
            per_video[base] = rec
            n_shard += 1
        shards.append((d.name, n_shard))
    return per_video, shards, dupes


def pct(xs):
    xs = [x for x in xs if x is not None]
    return 100 * float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops-dir", type=Path, required=True,
                    help="folder containing one sub-folder per output shard")
    ap.add_argument("--transition-times", type=Path, required=True,
                    help="Oops! transition_times.json")
    ap.add_argument("--shards", default="valALL",
                    help="substring of the shard folder names to include")
    ap.add_argument("--population", type=Path, default=None,
                    help="population JSON ('videos' with 'basename'), to report coverage")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write a per-video JSON (basename, pred_time, correctness)")
    args = ap.parse_args()

    per_video, shards, dupes = load_shards(args.oops_dir, args.shards)
    tt = json.load(open(args.transition_times, encoding="utf-8"))

    print("== shards included")
    for name, n in shards:
        print(f"   {name:24s} {n:5d} videos")
    print(f"   distinct videos: {len(per_video)}"
          + (f"  (duplicates dropped: {dupes})" if dupes else ""))

    rows, n_missing = [], 0
    for base, r in per_video.items():
        ann = tt.get(base)
        if ann is None:
            n_missing += 1
            continue
        raw_t = [t for t in (ann.get("t") or []) if t != -1]
        r = dict(r)
        r["raw_t"] = raw_t
        r["n_annot"] = len(raw_t)
        r["off25"] = official_correct(r["pred_time"], raw_t, 0.25)
        r["off1"] = official_correct(r["pred_time"], raw_t, 1.0)
        rows.append(r)
    if n_missing:
        print(f"   [!] {n_missing} videos without an official annotation: excluded")

    print()
    print("== OFFICIAL Oops! protocol (within 0.25 s / 1 s of any annotator)")
    print(f"   n = {len(rows)}")
    print(f"   Acc@0.25s = {pct([r['off25'] for r in rows]):.1f}%")
    print(f"   Acc@1s    = {pct([r['off1'] for r in rows]):.1f}%")

    cust25 = [r["custom"].get("accuracy_at_delta_0.25") for r in rows if r["custom"]]
    cust1 = [r["custom"].get("accuracy_at_delta_1") for r in rows if r["custom"]]
    iou_peak = [r["custom"].get("iou_peak") for r in rows if r["custom"]]
    cont = [r["custom"].get("contiguous_iou") for r in rows if r["custom"]]
    print()
    print("== metric saved by the released code")
    print(f"   n = {len(cust25)}")
    print(f"   Acc@0.25s (code GT)   = {pct(cust25):.1f}%")
    print(f"   Acc@1s    (code GT)   = {pct(cust1):.1f}%")
    print(f"   iou_peak              = {100*np.nanmean(np.array(iou_peak, dtype=float)):.1f}")
    print(f"   contiguous_iou        = {100*np.nanmean(np.array(cont, dtype=float)):.1f}")

    print()
    print("== effect of the ground-truth definition")
    print(f"   Acc@0.25s: {pct(cust25):.1f}% (code GT) -> {pct([r['off25'] for r in rows]):.1f}% (official)")
    print(f"   Acc@1s:    {pct(cust1):.1f}% (code GT) -> {pct([r['off1'] for r in rows]):.1f}% (official)")

    n_ann = [r["n_annot"] for r in rows]
    fps = [r["fps"] for r in rows]
    nfr = [r["n_frames_scored"] for r in rows]
    print()
    print("== configuration observed in the outputs")
    print(f"   surprise points per video: median {np.median(nfr):.0f} (min {min(nfr)}, max {max(nfr)})")
    print(f"   video fps: median {np.median(fps):.2f}")
    print(f"   annotators per video: mean {np.mean(n_ann):.2f}")

    if args.population is not None:
        pop = json.load(open(args.population, encoding="utf-8"))
        names = {v["basename"] for v in pop["videos"]}
        have = names & set(per_video)
        print()
        print(f"== population coverage ({args.population.name})")
        print(f"   {len(have)} / {len(names)} videos ({100*len(have)/len(names):.1f}%)")

    if args.dump is not None:
        out = [{"basename": r["basename"], "pred_time": r["pred_time"],
                "off25": r["off25"], "off1": r["off1"],
                "cust25": r["custom"].get("accuracy_at_delta_0.25"),
                "cust1": r["custom"].get("accuracy_at_delta_1"),
                "shard": r["shard"]} for r in rows]
        json.dump(out, open(args.dump, "w", encoding="utf-8"), indent=1)
        print(f"\nper-video results written to {args.dump}")


if __name__ == "__main__":
    main()
