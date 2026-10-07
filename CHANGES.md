# Changes with respect to upstream

This repository is a copy of [sahithyaravi/SPIKE-RL](https://github.com/sahithyaravi/SPIKE-RL)
at upstream commit **`13dcd93`** ("fix rope"), retrieved on 14 July 2026. The
full upstream history is preserved: every commit up to `13dcd93` is the
authors' own work, unchanged.

The upstream code is distributed under the **Apache License 2.0** (see
`LICENSE`), which this copy keeps. As required by section 4(b), modified files
carry a notice at the top, and every change is listed here.

## Modified files

| file | change |
|---|---|
| `src/open_r1_video/inference/inference.py` | adds the optional `--min_frames` argument, which replaces the per-dataset raw-frame floor. With the default (`None`) the behaviour is identical to upstream. |
| `README.md` | a short notice at the top pointing to this file. The rest of the authors' README is unchanged. |

## Added files

| file | purpose |
|---|---|
| `tools/make_oops_population.py` | builds the Oops! evaluation population from the official annotation files |
| `tools/build_spike_rl_input.py` | converts a population into the input format expected by `inference.py` |
| `tools/eval_spike_oops.py` | scores the outputs under the official Oops! protocol and under the metric saved by the code |
| `RUNNING.md` | how to run inference with this setup |
| `OUR_RESULTS.md` | the configuration we ran and the numbers we measured |
| `CHANGES.md` | this file |

No upstream file was deleted.
