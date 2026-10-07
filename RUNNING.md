# Running SPIKE on Oops!

How we run the released inference code on the Oops! validation set. Everything
uses the upstream method unchanged; the only addition is `--min_frames`, which
makes the frame budget explicit (see `CHANGES.md`).

## Requirements

- the environment described in the authors' README (Python 3.10, PyTorch, the
  packages in `setup.py`);
- the Oops! dataset (videos and annotations), from the official release by
  Epstein et al.;
- one GPU that fits Qwen2.5-VL-7B-Instruct in bf16.

## 1. Build the evaluation population

```bash
python3 tools/make_oops_population.py \
    --annotations /path/to/oops_dataset/annotations \
    --out spike_rl_input/oops_val_population.json
```

Rule: the dataset's filtered validation list (`val_filtered.txt`) restricted to
clips with at least one valid annotator label. On the released annotations this
gives **4,020 videos**.

## 2. Convert it to the layout `inference.py` expects

```bash
python3 tools/build_spike_rl_input.py \
    --subset spike_rl_input/oops_val_population.json \
    --video-root /path/to/oops_dataset/oops_video \
    --out-json spike_rl_input/oops_val.json \
    --out-video-root spike_rl_input/oops_val_videos
```

This writes the JSON in the loader's format and a symlink farm
`{basename}_merged/0_E_merged.mp4`. Nothing in the upstream loader is changed.

## 3. Run inference

```bash
python -u src/open_r1_video/inference/inference.py \
    --json_path  spike_rl_input/oops_val.json \
    --video_root spike_rl_input/oops_val_videos \
    --result_folder results/oops/valALL_budget8 \
    --method prior_frame_bayesian_approach \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --topk_hyp 3 \
    --min_frames 8
```

- `--min_frames 8` sets the frame budget. Omit it to use the upstream default
  for Oops! (32). Other values (16, 32, 64, ...) work the same way.
- `--num_entries N` runs a random subset of N videos, useful as a pilot.
- The script resumes: if `results.json` already exists in the result folder,
  completed videos are skipped.
- For large runs, split the population JSON into shards and give each its own
  `--result_folder` (we used folders named `valALL_budget8_p1a`, `..._p1b`, ...).
  `eval_spike_oops.py` merges every folder whose name contains `valALL`.

## 4. Score the outputs

```bash
python3 tools/eval_spike_oops.py \
    --oops-dir results/oops \
    --transition-times /path/to/oops_dataset/annotations/transition_times.json \
    --population spike_rl_input/oops_val_population.json \
    --dump results/oops/per_video.json
```

The script reports two numbers for each tolerance: under the official Oops!
protocol, and under the metric that `inference.py` saves. The peak selection is
the same in both; only the ground-truth definition differs.
