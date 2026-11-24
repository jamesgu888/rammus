"""
Runs the whole RAMMUS experiment.

Loads the MAFAULDA microphone recordings, scores every test clip with the RMS baseline and with RAMMUS
(both compared against all healthy recordings and against healthy recordings at a similar shaft speed),
prints the results, saves them to results/results.txt, and draws the figures.
"""

from pathlib import Path

import numpy as np

from data import cut_clips, fault_type, load_recordings, shaft_speed
from detectors import auroc, score_clips, split_healthy
from rammus import band_names, feature_names
from plots import plot_auroc_by_fault, plot_roc, plot_spectrogram, plot_spectrum, plot_speed_matching

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
seeds = [0, 1, 2, 3, 4]

report = []


def say(text=""):
    print(text)
    report.append(text)


# Load the microphone recordings and cut them into 1 s clips
names, signals = load_recordings()
clip_rms, clip_rammus, clip_recording = cut_clips(signals)
recording_speed = np.array([shaft_speed(n) for n in names])
recording_fault = np.array([fault_type(n) for n in names])
clip_fault = recording_fault[clip_recording]
faulty = clip_fault != "normal"
fault_types = sorted(set(clip_fault[faulty]))
healthy_recordings = np.flatnonzero(recording_fault == "normal")
say(f"{len(names)} recordings ({len(healthy_recordings)} healthy), {len(clip_fault)} clips")


def run(seed):
    # Score the test clips with both detectors, with and without speed matching.
    # Each alarm cutoff is the 95th percentile of the healthy calibration clips' scores.
    reference, calibration, healthy_test = split_healthy(healthy_recordings, seed)
    calibration_clips = np.flatnonzero(np.isin(clip_recording, calibration))
    test_clips = np.flatnonzero(np.isin(clip_recording, healthy_test) | faulty)

    scores = {"pooled": {}, "speed-matched": {}}
    cutoffs = {"pooled": {}, "speed-matched": {}}
    flagged = None
    for mode in scores:
        for name, values in [("RMS", clip_rms), ("RAMMUS", clip_rammus)]:
            matched = mode == "speed-matched"
            calibration_scores, _ = score_clips(values, calibration_clips, clip_recording, recording_speed, reference, matched)
            scores[mode][name], which = score_clips(values, test_clips, clip_recording, recording_speed, reference, matched)
            cutoffs[mode][name] = np.percentile(calibration_scores, 95)
            if matched and name == "RAMMUS":
                flagged = which
    return test_clips, scores, cutoffs, flagged


# Detailed results for the first random split
test_clips, scores, cutoffs, flagged = run(seeds[0])
test_faulty = faulty[test_clips]
test_fault = clip_fault[test_clips]
say(f"Test clips: {np.sum(~test_faulty)} healthy, {np.sum(test_faulty)} faulty")

for mode in scores:
    say(f"\n{mode}")
    for name in ["RMS", "RAMMUS"]:
        s = scores[mode][name]
        alarms = s > cutoffs[mode][name]
        caught = np.mean(alarms[test_faulty])
        false_alarms = np.mean(alarms[~test_faulty])
        # balanced accuracy weighs healthy and faulty clips equally (there are ~100 faulty clips per healthy one)
        balanced = (caught + 1 - false_alarms) / 2
        say(f"  {name:7s} caught {caught:.0%}, false alarms {false_alarms:.0%}, "
            f"balanced accuracy {balanced:.0%}, AUROC {auroc(s[~test_faulty], s[test_faulty]):.2f}")

say("\nBand that set off speed-matched RAMMUS on the faults it caught:")
caught_flags = flagged[test_faulty & (scores["speed-matched"]["RAMMUS"] > cutoffs["speed-matched"]["RAMMUS"])]
counts = np.bincount(caught_flags, minlength=len(feature_names))
for i in np.argsort(-counts):
    if counts[i] > 0:
        say(f"  {feature_names[i]:22s} {counts[i]}")

# AUROC over several random splits, overall and for each fault type
aurocs = {mode: {name: {} for name in ["RMS", "RAMMUS"]} for mode in ["pooled", "speed-matched"]}
for seed in seeds:
    seed_test, seed_scores, _, _ = run(seed)
    seed_faulty = faulty[seed_test]
    for mode in seed_scores:
        for name in seed_scores[mode]:
            s = seed_scores[mode][name]
            aurocs[mode][name].setdefault("all faults", []).append(auroc(s[~seed_faulty], s[seed_faulty]))
            for fault in fault_types:
                of_type = clip_fault[seed_test] == fault
                aurocs[mode][name].setdefault(fault, []).append(auroc(s[~seed_faulty], s[of_type]))

say(f"\nAUROC over {len(seeds)} random splits, mean [min, max]")
for mode in aurocs:
    say(f"  {mode}")
    for fault in ["all faults"] + fault_types:
        row = f"    {fault:24s}"
        for name in ["RMS", "RAMMUS"]:
            a = aurocs[mode][name][fault]
            row += f"  {name} {np.mean(a):.2f} [{np.min(a):.2f}, {np.max(a):.2f}]"
        say(row)

(results_dir / "results.txt").write_text("\n".join(report) + "\n")

# Figures
plot_roc([
    ("RAMMUS, speed-matched", scores["speed-matched"]["RAMMUS"], "tab:blue", "-"),
    ("RMS, speed-matched", scores["speed-matched"]["RMS"], "tab:gray", "-"),
    ("RAMMUS, pooled", scores["pooled"]["RAMMUS"], "tab:blue", "--"),
], test_faulty)
plot_auroc_by_fault(aurocs, fault_types)
plot_speed_matching(aurocs)

# A healthy recording next to a bearing outer race fault at about the same shaft speed
healthy = min(healthy_recordings, key=lambda r: abs(recording_speed[r] - 40))
outer_race = np.flatnonzero(recording_fault == "bearing outer_race")
fault = min(outer_race, key=lambda r: abs(recording_speed[r] - recording_speed[healthy]))
band = flagged[np.flatnonzero(clip_recording[test_clips] == fault)[0]] % len(band_names)
titles = [f"Healthy, shaft at {recording_speed[healthy]:.1f} Hz",
          f"Bearing outer race fault, shaft at {recording_speed[fault]:.1f} Hz"]
plot_spectrogram(signals[healthy].astype(float), signals[fault].astype(float), titles, band)
plot_spectrum(signals[healthy].astype(float), signals[fault].astype(float), titles, band)
