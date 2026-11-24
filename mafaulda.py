import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

data_dir = Path(__file__).parent / "data" / "mafaulda"
results_dir = Path(__file__).parent / "results" / "mafaulda"
data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# MAFAULDA machinery fault database, SMT/UFRJ, www02.smt.ufrj.br/~offshore/mfs/page_01.html
# A rotor rig recorded for 5 s at 50 kHz at many shaft speeds. Each recording is a CSV with 8 columns;
# the site lists them as: 1 tachometer, 2-4 underhang bearing accelerometer (axial, radial, tangential),
# 5-7 overhang bearing accelerometer (axial, radial, tangential), 8 microphone (Shure SM81, 20 Hz-20 kHz).
# We only use column 8, the microphone, because this project is about sound.
# The site has one zip per condition; we download 4 of the 6 (7.2 GB), not the 13 GB full set.
url = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
conditions = ["normal", "imbalance", "horizontal-misalignment", "overhang"]
microphone_column = 7  # column 8, counting from 0
recorded_rate = 50000

# About 90% of the microphone's power is one steady tone at 22.04 kHz, above the 20 kHz the microphone is
# rated for (probably the motor drive's switching frequency). It is the same in every condition and has
# nothing to do with the faults, but at 50 kHz it would fill the top octave band and set every band share.
# So we halve the sampling rate to 25 kHz; the anti-aliasing filter removes everything above 12.5 kHz,
# including the tone. The frames are still 1024 samples with a 512 hop (41 ms at 25 kHz, main.py: 64 ms),
# and the bands are octaves up to the new 12.5 kHz Nyquist frequency, the same pattern as main.py.
sample_rate = recorded_rate // 2
frame_len = 1024
hop = 512
window = np.hanning(frame_len)
f = np.fft.rfftfreq(frame_len, d=1/sample_rate)
band_edges = [0, 781.25, 1562.5, 3125, 6250, 12501]
band_names = ["0-0.8 kHz", "0.8-1.6 kHz", "1.6-3.1 kHz", "3.1-6.3 kHz", "6.3-12.5 kHz"]
feature_names = [f"median {b}" for b in band_names] + [f"95th pct {b}" for b in band_names]

# Each 5 s recording is cut into five 1 s clips (47 frames each).
# There are only 49 healthy recordings, so this gives 245 healthy clips to learn from and test on.
clip_len = sample_rate
seeds = [0, 1, 2, 3, 4]
# for the speed-matched detectors in section c: how many healthy reference recordings to compare against
neighbours = 8


def load_all():
    # Reads the microphone column of every CSV straight out of the zips (no unzipping),
    # resamples it to 25 kHz and caches the result, since parsing the CSVs takes a few minutes.
    cache = data_dir / "microphone_25k.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["names"], saved["signals"]
    names = []
    signals = []
    for condition in conditions:
        path = data_dir / f"{condition}.zip"
        if not path.exists():
            print(f"Downloading {condition}.zip (this can take several minutes)...")
            urlretrieve(url + f"{condition}.zip", path)
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
                with z.open(name) as csv:
                    s = np.loadtxt(csv, delimiter=",", usecols=microphone_column)
                names.append(name)
                signals.append(resample_poly(s, 1, 2).astype(np.float32))
        print(f"{condition}: {sum(n.startswith(condition) for n in names)} recordings")
    np.savez(cache, names=np.array(names), signals=np.array(signals))
    return np.array(names), np.array(signals)


def fault_class(name):
    # normal/12.288.csv -> "normal", imbalance/20g/xx.csv -> "imbalance",
    # overhang/ball_fault/6g/xx.csv -> "overhang ball_fault"
    parts = name.split("/")
    if parts[0] == "overhang":
        return f"overhang {parts[1]}"
    return parts[0]


def severity(name):
    # imbalance/20g/xx.csv -> "20g", horizontal-misalignment/1.5mm/xx.csv -> "1.5mm",
    # overhang/ball_fault/6g/xx.csv -> "6g" (an imbalance mass added on top of the bearing fault)
    return name.split("/")[-2]


def rotation_hz(name):
    # the file name is the shaft rotation frequency in Hz, e.g. 12.288.csv
    # (counting tachometer pulses gives 12.0 Hz for that file; they agree to within about 2.5%)
    return float(Path(name).stem)


def rms(s):
    return np.sqrt(np.mean(s ** 2))


def band_shares(s):
    # one row per 41 ms frame, one column per band; each row adds up to 1
    s = s - np.mean(s)
    shares = []
    for start in range(0, len(s) - frame_len + 1, hop):
        frame = s[start:start + frame_len] * window
        P = np.abs(np.fft.rfft(frame)) ** 2
        band_power = [np.sum(P[(f >= lo) & (f < hi)]) for lo, hi in zip(band_edges[:-1], band_edges[1:])]
        shares.append(np.array(band_power) / np.sum(P))
    return np.array(shares)


def rammus_features(s):
    shares = band_shares(s)
    return np.concatenate([np.median(shares, axis=0), np.percentile(shares, 95, axis=0)])


def auroc(healthy_scores, faulty_scores):
    # fraction of (faulty, healthy) pairs where the faulty clip scores higher; ties count half
    higher = faulty_scores[:, None] > healthy_scores[None, :]
    tied = faulty_scores[:, None] == healthy_scores[None, :]
    return np.mean(higher) + 0.5 * np.mean(tied)


# Same scores as main.py, except the healthy reference clips are passed in instead of being globals,
# so that each random split (and each shaft speed in section c) can refit them.
def rms_score(reference_rms, rms_values):
    return np.abs(rms_values - np.mean(reference_rms)) / np.std(reference_rms)


def rammus_score(reference_features, features):
    feature_mean = np.mean(reference_features, axis=0)
    feature_std = np.std(reference_features, axis=0)
    keep = feature_std > 0
    z = np.abs(features[:, keep] - feature_mean[keep]) / feature_std[keep]
    return np.max(z, axis=1), np.flatnonzero(keep)[np.argmax(z, axis=1)]


# a: load the recordings and cut them into 1 s clips
names, signals = load_all()
clip_rms = []
clip_features = []
clip_recording = []
for i, s in enumerate(signals):
    for start in range(0, len(s) - clip_len + 1, clip_len):
        clip = s[start:start + clip_len].astype(float)
        clip_rms.append(rms(clip))
        clip_features.append(rammus_features(clip))
        clip_recording.append(i)
clip_rms = np.array(clip_rms)
clip_features = np.array(clip_features)
clip_recording = np.array(clip_recording)
recording_speed = np.array([rotation_hz(n) for n in names])
clip_speed = recording_speed[clip_recording]
clip_class = np.array([fault_class(names[i]) for i in clip_recording])
clip_severity = np.array([severity(names[i]) for i in clip_recording])
faulty = clip_class != "normal"
fault_classes = [c for c in dict.fromkeys(clip_class) if c != "normal"]
print(f"{len(names)} recordings, {np.sum(~faulty)} healthy clips, {np.sum(faulty)} faulty clips, 1 s each")
lines = ["Recordings per condition (shaft speed range):"]
for c in ["normal"] + fault_classes:
    speeds = recording_speed[[fault_class(n) == c for n in names]]
    lines.append(f"  {c:24s} {len(speeds):4d}  ({speeds.min():.1f}-{speeds.max():.1f} Hz)")


# b: split the healthy recordings (not clips) 60/20/20 into reference / calibration / test,
# so a healthy test clip never comes from a recording the detectors learned from
def split(seed):
    rng = np.random.default_rng(seed)
    healthy_recordings = np.flatnonzero(np.array([fault_class(n) == "normal" for n in names]))
    rng.shuffle(healthy_recordings)
    n_reference = int(0.6 * len(healthy_recordings))
    n_calibration = int(0.2 * len(healthy_recordings))
    group = np.full(len(faulty), "test", dtype=object)  # every faulty clip is a test clip
    group[np.isin(clip_recording, healthy_recordings[:n_reference])] = "reference"
    group[np.isin(clip_recording, healthy_recordings[n_reference:n_reference + n_calibration])] = "calibration"
    return group


# c: score clips. "pooled" is exactly main.py: one healthy mean and spread from all reference clips.
# But the healthy recordings run from 12 to 61 Hz, and the sound changes a lot with speed
# (every RAMMUS measurement has |correlation| > 0.7 with shaft speed on healthy clips), so the pooled
# spread is mostly speed. "speed-matched" compares each clip only with the reference clips from the
# 8 healthy reference recordings closest to it in shaft speed (40 clips), the continuous version of main.py
# splitting by machine setting. The shaft speed comes from the tachometer, which a real monitor would have.
def scores(name, group, clips, matched):
    values = clip_rms if name == "RMS" else clip_features
    score = rms_score if name == "RMS" else lambda r, v: rammus_score(r, v)[0]
    reference = group == "reference"
    if not matched:
        return score(values[reference], values[clips])
    reference_recordings = np.unique(clip_recording[reference])
    out = []
    for i in np.flatnonzero(clips):
        distance = np.abs(recording_speed[reference_recordings] - clip_speed[i])
        nearest = reference_recordings[np.argsort(distance)[:neighbours]]
        out.append(score(values[np.isin(clip_recording, nearest)], values[i:i + 1])[0])
    return np.array(out)


# fit on the reference clips, put each cutoff at the 95th percentile of the calibration clips' scores,
# then score the test clips
def evaluate(group, matched):
    results = {}
    for name in ["RMS", "RAMMUS"]:
        cutoff = np.percentile(scores(name, group, group == "calibration", matched), 95)
        test_scores = scores(name, group, group == "test", matched)
        results[name] = (test_scores, test_scores > cutoff, cutoff)
    return results


group = split(seeds[0])
test = group == "test"
is_faulty = faulty[test]
lines.append(f"\nSplit of the {len(set(clip_recording[~faulty]))} healthy recordings (seed {seeds[0]}): "
             f"{len(set(clip_recording[group == 'reference']))} reference, "
             f"{len(set(clip_recording[group == 'calibration']))} calibration, "
             f"{len(set(clip_recording[~faulty & test]))} test")
all_results = {}
for matched in [False, True]:
    results = evaluate(group, matched)
    all_results[matched] = results
    title = "speed-matched" if matched else "pooled, as in main.py"
    lines.append(f"\nTest, {title} ({np.sum(~is_faulty)} healthy, {np.sum(is_faulty)} faulty clips)")
    for name in ["RMS", "RAMMUS"]:
        test_scores, alarms, cutoff = results[name]
        caught = np.sum(alarms & is_faulty)
        passed = np.sum(~alarms & ~is_faulty)
        # plain accuracy is dominated by the many faulty clips, so also give the balanced accuracy:
        # the average of the catch rate and the pass rate, i.e. the accuracy if both groups were the same size
        accuracy = (caught + passed) / len(test_scores)
        balanced = (caught / np.sum(is_faulty) + passed / np.sum(~is_faulty)) / 2
        lines.append(
            f"  {name:7s} caught {caught:4d}/{np.sum(is_faulty)} faults, "
            f"false alarms {np.sum(alarms & ~is_faulty):2d}/{np.sum(~is_faulty)}, "
            f"accuracy {accuracy:.3f} (balanced {balanced:.3f}), "
            f"AUROC {auroc(test_scores[~is_faulty], test_scores[is_faulty]):.3f}, cutoff {cutoff:.1f}"
        )
    lines.append("  By fault class:              caught RMS  caught RAMMUS  AUROC RMS  AUROC RAMMUS")
    for c in fault_classes:
        of_class = clip_class[test] == c
        row = f"    {c:26s}"
        for name in ["RMS", "RAMMUS"]:
            row += f" {np.sum(results[name][1] & of_class):4d}/{np.sum(of_class):<4d}  "
        for name in ["RMS", "RAMMUS"]:
            test_scores = results[name][0]
            row += f"    {auroc(test_scores[~is_faulty], test_scores[of_class]):.3f}  "
        lines.append(row.rstrip())

# The AUROC of each fault severity, for the speed-matched detectors.
# The zips' file dates suggest normal and imbalance 6-25 g were recorded on one day (2014-10-03) and
# imbalance 30-35 g and all misalignment on another (2014-10-06), so watch for a jump between 25 g and 30 g.
lines.append("\nSpeed-matched AUROC by fault severity (seed 0):  RMS    RAMMUS")
for c in fault_classes:
    for v in sorted(set(clip_severity[clip_class == c]), key=lambda v: float(v.rstrip("gm"))):
        of_severity = (clip_class[test] == c) & (clip_severity[test] == v)
        row = f"    {c:26s} {v:6s}"
        for name in ["RMS", "RAMMUS"]:
            test_scores = all_results[True][name][0]
            row += f"    {auroc(test_scores[~is_faulty], test_scores[of_severity]):.3f}"
        lines.append(row)

# d: repeat with other random splits of the healthy recordings to see how much the AUROC depends on the split
aurocs = {}
for seed in seeds:
    seed_group = split(seed)
    seed_test = seed_group == "test"
    seed_faulty = faulty[seed_test]
    for matched in [False, True]:
        results = evaluate(seed_group, matched)
        for c in ["all faults"] + fault_classes:
            of_class = seed_faulty if c == "all faults" else clip_class[seed_test] == c
            for name in ["RMS", "RAMMUS"]:
                test_scores = results[name][0]
                aurocs.setdefault((matched, c, name), []).append(auroc(test_scores[~seed_faulty], test_scores[of_class]))
for matched in [False, True]:
    title = "speed-matched" if matched else "pooled"
    lines.append(f"\nAUROC over {len(seeds)} random splits of the healthy recordings "
                 f"(seeds {seeds[0]}-{seeds[-1]}), {title}: mean [min, max]")
    lines.append("                                RMS                     RAMMUS")
    for c in ["all faults"] + fault_classes:
        row = f"  {c:26s}"
        for name in ["RMS", "RAMMUS"]:
            a = aurocs[(matched, c, name)]
            row += f"    {np.mean(a):.3f} [{np.min(a):.3f}, {np.max(a):.3f}]"
        lines.append(row)

print("\n".join(lines))
(results_dir / "results.txt").write_text("\n".join(lines).strip() + "\n")

# e: score histograms for the seed 0 split, pooled on top and speed-matched below, drawn like cwru.py
# Faulty scores reach the thousands, so the x axis is logarithmic (scores below 0.1 go in the first bin);
# the scores above 30 are mostly imbalance 30-35 g, where the microphone records a loud knock every revolution.
# There are about 100 times more faulty clips than healthy ones, so each group's bars show its fraction of clips.
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
max_score = max(all_results[matched][name][0].max() for matched in [False, True] for name in ["RMS", "RAMMUS"])
bins = np.geomspace(0.1, max_score * 1.1, 40)
for row, matched in enumerate([False, True]):
    for ax, name in [(axes[row, 0], "RMS"), (axes[row, 1], "RAMMUS")]:
        test_scores, alarms, cutoff = all_results[matched][name]
        for label, s in [("Healthy", test_scores[~is_faulty]), ("Faulty", test_scores[is_faulty])]:
            s = np.clip(s, 0.1, None)
            ax.hist(s, bins=bins, weights=np.ones(len(s)) / len(s), alpha=0.6, label=label)
        ax.axvline(cutoff, color="black", linestyle="--", label="Cutoff")
        ax.set_xscale("log")
        ax.set_title(f"{name} scores, {'speed-matched' if matched else 'pooled (as in main.py)'}")
        if matched:
            ax.set_xlabel("Score (standard deviations from healthy)")
        ax.set_ylabel("Fraction of clips")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # room for the legend above the bars
        ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(results_dir / "scores.png", dpi=150)
plt.show()
