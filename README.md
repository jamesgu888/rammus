# RAMMUS

Rotational Anomaly Monitoring in Motors Using Sound.

Can we tell from a ten-second recording that a motor bearing is running off-center?
This project compares two detectors that learn only from healthy bearings.

1. **RMS amplitude detector (baseline).** One number per clip, the root-mean-square amplitude, compared against healthy clips.
2. **RAMMUS.** Short FFTs, the share of energy in five frequency bands (0-0.5, 0.5-1, 1-2, 2-4, 4-8 kHz), summarized per clip by the median and 95th percentile of each share.
   A clip is flagged when any of these ten measurements is unusually far from its healthy average.

See [docs/proposal.md](docs/proposal.md) for the plan and [docs/research.md](docs/research.md) for background research.

## Data

MIMII DG bearing recordings (Dohi et al., DCASE Workshop 2022), section 00.
Download `bearing.zip` (772 MB) from [Zenodo record 6529888](https://zenodo.org/records/6529888) and unzip it into `data/`, so the clips are in `data/bearing/train` and `data/bearing/test`.

```sh
curl -L -o data/bearing.zip "https://zenodo.org/records/6529888/files/bearing.zip?download=1"
unzip data/bearing.zip -d data
```

## Run

```sh
python -m pip install -r requirements.txt
python main.py
```

Close each plot window to continue to the next step.
The script writes `results/results.txt`, `results/split.txt`, `results/scores.png` and `results/spectrogram.png`.

## Authors

James Gu and Tristan Zhu, Stanford EE102A.
