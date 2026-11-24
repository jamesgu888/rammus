# RAMMUS

Rotational Anomaly Monitoring in Motors Using Sound.

Can we hear a faulty bearing with a microphone?
This project compares two detectors that learn only from healthy recordings.

1. **RMS amplitude detector (baseline).** One number per clip, the root-mean-square amplitude, compared against healthy clips.
2. **RAMMUS.** Short FFTs, the share of energy in five octave bands, summarized per clip by the median and 95th percentile of each share.
   A clip is flagged when any of these ten measurements is unusually far from healthy.

Both detectors compare each clip with healthy recordings at a similar shaft speed, because a healthy machine sounds different at different speeds.

See [docs/proposal.md](docs/proposal.md) for the original proposal, which used the MIMII DG dataset.

## Data

[MAFAULDA](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html) (Machinery Fault Database, UFRJ): a motor turning a shaft on two bearings, recorded for 5 s at 50 kHz at shaft speeds from 12 to 61 Hz.
We use only the microphone channel (column 8 of each CSV).
We use four conditions: normal (49 recordings), imbalance, horizontal misalignment, and overhang bearing faults (ball, cage, outer race).

`main.py` downloads the four zips (about 7 GB) the first time, saves just the microphone signal to `data/mafaulda_microphone_25k.npz` (521 MB), and deletes the zips.

## Method details

- The microphone picks up a loud 22 kHz tone from the motor drive, so the audio is downsampled to 25 kHz, which removes it.
- Each 5 s recording becomes five 1 s clips.
- FFT frames are 1024 samples (41 ms) with a Hann window and a 512-sample hop.
- Bands: 0-0.8, 0.8-1.6, 1.6-3.1, 3.1-6.3 and 6.3-12.5 kHz.
- Healthy recordings are split 60/20/20 into reference, calibration and test by whole recording, so no test clip comes from a recording the detectors learned from.
- Each clip is compared with the 8 healthy reference recordings closest to its shaft speed.
- The alarm cutoff is the 95th percentile of healthy calibration scores.

## Run

```sh
python -m pip install -r requirements.txt
python main.py
python plots.py
```

`main.py` prints the results and writes `results/results.txt`, `results/scores.csv` and `results/auroc_by_seed.csv`.
`plots.py` draws the figures in `results/` from those files.

## Results

AUROC, mean over 5 random splits of the healthy recordings (0.5 is a coin flip, 1.0 is perfect):

| | RMS | RAMMUS |
| --- | ---: | ---: |
| Compared with all healthy recordings | 0.66 | 0.73 |
| Compared with healthy recordings at a similar speed | 0.74 | **0.85** |

By fault type (speed-matched): bearing cage 0.91, bearing outer race 0.90, misalignment 0.93, imbalance 0.80, bearing ball 0.69.
RMS only wins on ball faults.

## Caveats

- There are only 49 healthy recordings, so each test uses 11 of them, and the confidence interval on the RAMMUS - RMS difference for a single split is wide.
- Some heavy imbalance recordings have a loud knock every revolution that any detector catches.
- File dates suggest misalignment was recorded on a different day from the healthy recordings, which may make it easier to tell apart.

## Authors

James Gu and Tristan Zhu, Stanford EE102A.
