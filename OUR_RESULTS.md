# Our runs: configuration and results

What we ran with this code, and what we measured. The per-video outputs are
available as a separate download (see the repository's Releases), so every
number below can be recomputed with `tools/eval_spike_oops.py` without running
inference again.

## Configuration

| | value |
|---|---|
| method | `prior_frame_bayesian_approach` |
| model | `Qwen/Qwen2.5-VL-7B-Instruct` (base model) |
| hypotheses per step | `--topk_hyp 3` |
| frame budget | `--min_frames 8`: 8 scoring points per video (verified in the outputs) |
| population | Oops! validation, 4,020 videos (see `RUNNING.md`, step 1) |
| coverage | 4,020 / 4,020 videos, 7 disjoint shards, no duplicates |
| median clip length | 7.9 s |

**Which variant this is.** We ran the inference-time method with the base
model. The RL-trained weights are not part of the release, so these are
**SPIKE** results, not SPIKE-RL.

**On the frame budget.** Three values appear in the sources, and we report
which one we used:

- the paper, App. C.2: a base budget of 8 frames for clips up to one minute;
- the paper, Table 8: 64 is referred to as the default budget;
- the released code: a floor of `min_frames = 32` for Oops!.

We ran with 8, the value given by the duration rule of App. C.2 for clips of
this length. Runs at 32 and 64 use the same command with a different
`--min_frames`.

## Results on the 4,020 videos

Peak selection is identical in all rows (argmax of `surprise_scores`); only
the ground-truth definition changes.

| ground truth | Acc@0.25s | Acc@1s |
|---|---|---|
| official Oops! protocol: within tolerance of any annotator label | **12.6** | **33.5** |
| metric saved by `inference.py`: sampled frames inside `[0.8·t, 0.8·duration]` | 37.6 | 49.0 |

Also saved by the code: `iou_peak` 39.4, `contiguous_iou` 94.3.

**On the two definitions.** The official protocol (Epstein et al., CVPR 2020,
Sec. 5.3) counts a prediction as correct if it lies within 0.25 s or 1 s of any
of the up-to-three annotated transition times. The metric saved by the released
code uses the window `[0.8·t, 0.8·duration]`, restricted to the frames that
were sampled; at 8 frames per clip this window contains on average 37% of the
sampled points. In 208 of the 4,020 videos (5.2%) no sampled frame falls inside
the window, and the code assigns 0 to every metric for that video.

## Stability of the subset

The first 3,000 videos and the remaining 1,020 give consistent numbers under
the official protocol (12.5 / 33.7 and 12.9 / 32.6), so the result does not
depend on which shard a video fell into.
