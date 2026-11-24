# Research notes

Background research done before implementation, November 2026.
Sources are linked inline; treat numbers from technical reports as approximate until re-checked.

## Dataset facts (MIMII DG bearing, section 00)

Sources: [Zenodo 6529888](https://zenodo.org/records/6529888), [MIMII DG paper, arXiv 2205.13879](https://arxiv.org/abs/2205.13879) Table 1, [DCASE 2022 Task 2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring).

- Clips are mono, 16 kHz, 16-bit, 10 s.
  Factory noise is mixed in at +12 dB SNR.
- Filename pattern: `section_00_{source|target}_{train|test}_{normal|anomaly}_{NNNN}_vel_{v}.wav`, where `vel` is shaft speed in krpm.
  The same information is in `bearing/attributes_00.csv`.
- Download: `bearing.zip` from Zenodo 6529888 (772 MB), identical to `dev_bearing.zip` in the DCASE dev record 6355122.
- Source train: 990 normal clips, 198 at each of 6, 10, 14, 18, 22 krpm.
- Target train: 10 normal clips at 8 (2), 12 (3), 16 (3), 20 (2) krpm.
- Source test: 50 normal + 50 anomaly, 10 of each per source speed.
- Target test: 50 normal + 50 anomaly across 8 speeds: 2, 4, 8, 12, 16, 20, 24, 26 krpm, with 6 to 8 clips per class per speed.
- No target speed equals a source speed.
  2, 4, 24, 26 krpm are outside the training range; 8 to 20 krpm fall between training speeds and also appear in the 10 target train clips.
- Anomaly type is shaft eccentricity in two directions; filenames do not say which direction.
- The only domain shift in section 00 is speed.
  Section 01 varies microphone location and section 02 varies noise type.

### Corrections to the proposal

- "8 unseen speeds" should say that only 4 are outside the training range, and that 4 others appear in the excluded target train clips.
- Report target results grouped (interpolated vs extrapolated speeds), not per speed: 6 to 8 clips per class per speed is too few.
- "Off-center" is fine as a paraphrase of eccentricity, but mention there are two directions.

## Reference numbers

Bearing section 00, dev set, from the DCASE page.
The task-description paper (arXiv 2206.05876) has an error in its AE table for bearing; do not cite it for these.

| Baseline | AUC source | AUC target | pAUC |
| --- | --- | --- | --- |
| Autoencoder | 57.48 ± 3.66 | 63.07 ± 3.01 | 51.49 ± 1.07 |
| MobileNetV2 | 67.85 ± 19.61 | 60.17 ± 7.24 | 54.41 ± 5.72 |

DCASE metric definitions, if we quote these:

- Per-domain AUC uses that domain's normal clips against all anomalous clips in the section.
- pAUC uses FPR in [0, 0.1], pools both domains, and is McClish-standardized: `roc_auc_score(y, s, max_fpr=0.1)`, where 0.5 is chance.
- Official score is the harmonic mean of AUC source, AUC target and pAUC.
- See the [baseline evaluation code](https://github.com/Kota-Dohi/dcase2022_task2_baseline_ae/blob/main/01_test.py).

## What prior work says about bearing

- Bearing is the hardest machine type in MIMII DG; strong DCASE 2022 systems scored about 52 to 65 AUC on bearing source while scoring 75 to 99 on other types ([Morita](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Morita_45_t2.pdf), [Wilkinghoff](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Wilkinghoff_2_t2.pdf)).
- The closest prior method to RAMMUS (time-pooled log-mel plus GMM, [Guan](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Guan_24_t2.pdf)) got 60.2 source and 87.4 target AUC on bearing.
- The useful signal looks tonal and low-frequency: Morita used the harmonic component for bearing, and the first-place team ([Liu](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf)) turned off high-pass filtering for bearing.
- Eccentricity should show up at the shaft rate and its harmonics, not as the impulsive race-defect pattern that envelope analysis targets.
- Expectation: modest AUCs, with most of the signal below 2 kHz.
  Run-to-run variance on bearing is large (MobileNetV2 ±19.6), so any single number is noisy.

## Weaknesses in the plan, ranked, with fixes

1. **Speed confound (biggest issue).**
   Pooling all speeds into one healthy reference widens the standard deviations and hides faults.
   At new speeds, harmonics move across band edges and cause false alarms.
   Fix: split 792/198 stratified by speed and z-score against same-speed healthy clips.
   For target speeds, regress each feature on speed using source data and z-score the residual.
   Optional signals exercise: estimate shaft speed from the spectral peak instead of reading it from the filename.
2. **Small test sets.**
   At AUC 0.7 with 50/50 clips the 95% CI is about ±0.10, and a false-alarm count of 3/50 has a Clopper-Pearson interval of about 1 to 17%.
   Fix: freeze the design before looking at test labels (or tune on section 01), and report Clopper-Pearson CIs for every rate.
3. **Bounded share features.**
   Shares are skewed, sum to 1, and the 95th percentile has heavy tails.
   Fix: use 10·log10(share), i.e. band level in dB relative to total, with a robust z (median and MAD).
4. **No window.**
   A rectangular window has -13 dB sidelobes, so strong low-frequency harmonics leak into the quiet 4 to 8 kHz band.
   Fix: Hann window, drop the DC bin, half-open band edges with the Nyquist bin in the top band.
5. **Noise.**
   Shares are invariant to gain but not to SNR, and the 95th-percentile features can fire on factory-noise transients.
   Fix: say so, and check which feature fires on healthy false alarms.
6. **max |z| over correlated features.**
   False-alarm control is fine because the threshold is calibrated empirically.
   The cost is lost power from ignoring joint shifts.
   Keep it for interpretability and add Mahalanobis distance on the log shares (sklearn `LedoitWolf`) as an ablation.
7. **95th-percentile threshold.**
   About 5% false alarms on healthy data is by design, not a flaw.
   The informative results are detection rate at that point and how much the false-alarm rate grows under speed shift.
   Use `np.quantile(..., method="higher")`.
8. **Low bands are too coarse.**
   0 to 500 Hz lumps all shaft harmonics together; consider splitting into 0 to 250 and 250 to 500 Hz.
9. **RMS baseline** is also speed-confounded; compute it in dB and keep two-sided |z|.

## Implementation plan

- Load with `soundfile.read(path, dtype="float32")`; avoid `librosa.load` resampling to 22.05 kHz by default.
- Write the STFT by hand (`sliding_window_view`, Hann, `np.fft.rfft`) and unit-test it against `scipy.signal.ShortTimeFFT`.
- Bootstrap: B = 10,000 with `np.random.default_rng(seed)`, resample healthy and faulty indices separately with replacement, apply the same indices to both detectors, take the 2.5 and 97.5 percentiles of the AUC difference.
  Cross-check with DeLong.
- Write `splits.csv` (filename, split, speed, label), the seed, package versions and the git hash into each results directory.
- Report pAUC alongside AUC.

Planned layout:

```
data/        (gitignored) raw/ + splits.csv
src/rammus/  io.py stft.py features.py score.py eval.py
scripts/     make_splits.py extract.py evaluate.py
notebooks/   exploration only, importing from src
results/     <run_id>/ metrics.json, scores.csv, figs/
tests/       stft and feature unit tests
```
