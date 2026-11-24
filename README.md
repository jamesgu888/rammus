# RAMMUS

Rotational Anomaly Monitoring in Motors Using Sound.

Can we tell from a ten-second recording that a motor bearing is running off-center?
This project compares two unsupervised detectors that learn only from healthy bearings.

1. **RMS amplitude detector (baseline).** One number per clip, the root-mean-square amplitude, z-scored against healthy clips.
2. **RAMMUS.** Short FFTs, the share of energy in five frequency bands (0-0.5, 0.5-1, 1-2, 2-4, 4-8 kHz), summarized per clip by the median and 95th percentile of each share.
   A clip is flagged when any of these ten features departs unusually far from its healthy average.

See [docs/proposal.md](docs/proposal.md) for the full plan.

## Data

MIMII DG bearing recordings (Dohi et al., DCASE Workshop 2022, Zenodo record 6529888), section 00.
Audio is not committed; download it into `data/`.

## Layout

```
src/rammus/   feature extraction, detectors, evaluation
scripts/      runnable entry points (download, split, run experiments)
notebooks/    exploration and figures
results/      generated metrics and plots
data/         raw audio (gitignored)
docs/         proposal and write-up
```

## Setup

```
uv sync
```

## Authors

James Gu and Tristan Zhu, Stanford EE102A.
