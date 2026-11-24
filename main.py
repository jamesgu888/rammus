import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.signal import resample_poly

data_dir = Path(__file__).parent / "data"
results_dir = Path(__file__).parent / "results"
data_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# MAFAULDA machinery fault database (www02.smt.ufrj.br/~offshore/mfs/page_01.html):
# a motor turning a shaft on two bearings, recorded for 5 s at 50 kHz at many shaft speeds.
# Each recording is a CSV with 8 columns; column 8 is a microphone, the only one we use.
url = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
conditions = ["normal", "imbalance", "horizontal-misalignment", "overhang"]
microphone_column = 7  # column 8, counting from 0

# The microphone also picks up a loud 22 kHz tone from the motor drive in every recording.
# Halving the sample rate to 25 kHz filters it out, so the highest frequency we analyze is 12.5 kHz.
sample_rate = 25000
frame_len = 1024  # 41 ms per FFT
hop = 512
window = np.hanning(frame_len)
f = np.fft.rfftfreq(frame_len, d=1/sample_rate)
band_edges = [0, 781.25, 1562.5, 3125, 6250, 12501]
band_names = ["0-0.8 kHz", "0.8-1.6 kHz", "1.6-3.1 kHz", "3.1-6.3 kHz", "6.3-12.5 kHz"]
feature_names = [f"median {b}" for b in band_names] + [f"95th pct {b}" for b in band_names]

clip_len = sample_rate  # each 5 s recording is cut into five 1 s clips
neighbours = 8  # a clip is compared with the 8 healthy reference recordings closest to its shaft speed
seeds = [0, 1, 2, 3, 4]


def load_recordings():
    # Reading the CSVs takes a few minutes, so the microphone signals are saved to one file the first time.
    cache = data_dir / "mafaulda_microphone_25k.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["names"], saved["signals"]
    names = []
    signals = []
    for condition in conditions:
        path = data_dir / f"{condition}.zip"
        if not path.exists():
            print(f"Downloading {condition}.zip (several minutes)...")
            urlretrieve(url + f"{condition}.zip", path)
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
                with z.open(name) as csv:
                    s = np.loadtxt(csv, delimiter=",", usecols=microphone_column)
                names.append(name)
                signals.append(resample_poly(s, 1, 2).astype(np.float32))
        path.unlink()  # the zips are 7 GB in total; once the microphone column is saved we don't need them
    np.savez(cache, names=np.array(names), signals=np.array(signals))
    return np.array(names), np.array(signals)


def fault_class(name):
    # normal/12.288.csv -> "normal", imbalance/20g/xx.csv -> "imbalance",
    # overhang/ball_fault/6g/xx.csv -> "bearing ball_fault"
    parts = name.split("/")
    if parts[0] == "overhang":
        return f"bearing {parts[1]}"
    return parts[0]


def shaft_speed(name):
    # the file name is the shaft rotation frequency in Hz, e.g. 12.288.csv
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


def z_score(reference, value):
    # how many standard deviations each measurement is from the healthy reference; the biggest one is the score
    mean = np.mean(reference, axis=0)
    std = np.std(reference, axis=0)
    keep = std > 0
    z = np.abs(value[keep] - mean[keep]) / std[keep]
    return np.max(z), np.flatnonzero(keep)[np.argmax(z)]


def auroc(healthy_scores, faulty_scores):
    # fraction of (faulty, healthy) pairs where the faulty clip scores higher; ties count half
    higher = faulty_scores[:, None] > healthy_scores[None, :]
    tied = faulty_scores[:, None] == healthy_scores[None, :]
    return np.mean(higher) + 0.5 * np.mean(tied)


# a: load the recordings and cut them into 1 s clips
names, signals = load_recordings()
clip_rms = []
clip_features = []
clip_recording = []
for i, s in enumerate(signals):
    for start in range(0, len(s) - clip_len + 1, clip_len):
        clip = s[start:start + clip_len].astype(float)
        clip_rms.append([rms(clip)])
        clip_features.append(rammus_features(clip))
        clip_recording.append(i)
clip_rms = np.array(clip_rms)
clip_features = np.array(clip_features)
clip_recording = np.array(clip_recording)
recording_speed = np.array([shaft_speed(n) for n in names])
clip_speed = recording_speed[clip_recording]
clip_class = np.array([fault_class(names[i]) for i in clip_recording])
faulty = clip_class != "normal"
fault_classes = sorted(set(clip_class[faulty]))
healthy_recordings = np.flatnonzero([fault_class(n) == "normal" for n in names])
print(f"{len(names)} recordings ({len(healthy_recordings)} healthy), {len(clip_class)} clips")


# b: split the healthy recordings 60/20/20 into reference / calibration / test.
# We split whole recordings, so a healthy test clip never comes from a recording the detectors learned from.
def split(seed):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(healthy_recordings)
    n_reference = int(0.6 * len(shuffled))
    n_calibration = int(0.2 * len(shuffled))
    return shuffled[:n_reference], shuffled[n_reference:n_reference + n_calibration], shuffled[n_reference + n_calibration:]


# c: score clips against the healthy reference.
# "pooled" compares every clip with all reference recordings, which is what the proposal describes.
# "speed-matched" only uses the reference recordings closest in shaft speed, because the sound of a
# healthy machine changes a lot with speed (healthy recordings run from 12 to 61 Hz).
def detector_scores(values, clips, reference_recordings, speed_matched):
    scores = []
    flagged = []
    for i in clips:
        use = reference_recordings
        if speed_matched:
            distance = np.abs(recording_speed[reference_recordings] - clip_speed[i])
            use = reference_recordings[np.argsort(distance)[:neighbours]]
        score, feature = z_score(values[np.isin(clip_recording, use)], values[i])
        scores.append(score)
        flagged.append(feature)
    return np.array(scores), np.array(flagged)


def run(seed):
    reference, calibration, healthy_test = split(seed)
    calibration_clips = np.flatnonzero(np.isin(clip_recording, calibration))
    test_clips = np.flatnonzero(np.isin(clip_recording, healthy_test) | faulty)
    results = {}
    for mode in ["pooled", "speed-matched"]:
        for name, values in [("RMS", clip_rms), ("RAMMUS", clip_features)]:
            calibration_scores = detector_scores(values, calibration_clips, reference, mode == "speed-matched")[0]
            cutoff = np.percentile(calibration_scores, 95)
            scores, flagged = detector_scores(values, test_clips, reference, mode == "speed-matched")
            results[mode, name] = (scores, flagged, cutoff)
    return test_clips, results


# d: results for the first random split
test_clips, results = run(seeds[0])
is_faulty = faulty[test_clips]
test_class = clip_class[test_clips]
lines = [f"Test clips: {np.sum(~is_faulty)} healthy, {np.sum(is_faulty)} faulty (seed {seeds[0]})"]
for mode in ["pooled", "speed-matched"]:
    lines.append(f"\n{mode}")
    for name in ["RMS", "RAMMUS"]:
        scores, flagged, cutoff = results[mode, name]
        alarms = scores > cutoff
        caught = np.mean(alarms[is_faulty])
        false_alarms = np.mean(alarms[~is_faulty])
        # there are about 100 faulty clips per healthy one, so plain accuracy would just be the catch rate;
        # balanced accuracy weighs healthy and faulty clips equally
        balanced_accuracy = (caught + 1 - false_alarms) / 2
        lines.append(
            f"  {name:7s} caught {caught:4.0%} of faults, false alarms {false_alarms:4.0%}, "
            f"balanced accuracy {balanced_accuracy:.0%}, AUROC {auroc(scores[~is_faulty], scores[is_faulty]):.2f}"
        )
    lines.append("  AUROC by fault type:        RMS  RAMMUS")
    for c in fault_classes:
        row = f"    {c:24s}"
        for name in ["RMS", "RAMMUS"]:
            scores = results[mode, name][0]
            row += f"  {auroc(scores[~is_faulty], scores[test_class == c]):.2f}"
        lines.append(row)

scores, flagged, cutoff = results["speed-matched", "RAMMUS"]
caught_flags = flagged[is_faulty & (scores > cutoff)]
lines.append("\nMeasurement that set off speed-matched RAMMUS on the faults it caught:")
for i in np.argsort(-np.bincount(caught_flags, minlength=len(feature_names))):
    if np.sum(caught_flags == i) > 0:
        lines.append(f"  {feature_names[i]:22s} {np.sum(caught_flags == i)}")

# e: 95% confidence interval on the AUROC difference (speed-matched RAMMUS - RMS), by resampling
# whole recordings with replacement, since the five clips of one recording are not independent
rng = np.random.default_rng(seeds[0])
rammus_scores = results["speed-matched", "RAMMUS"][0]
rms_scores = results["speed-matched", "RMS"][0]
test_recording = clip_recording[test_clips]
healthy_recs = np.unique(test_recording[~is_faulty])
faulty_recs = np.unique(test_recording[is_faulty])
differences = []
for _ in range(1000):
    h = np.concatenate([np.flatnonzero(test_recording == r) for r in rng.choice(healthy_recs, len(healthy_recs))])
    a = np.concatenate([np.flatnonzero(test_recording == r) for r in rng.choice(faulty_recs, len(faulty_recs))])
    differences.append(auroc(rammus_scores[h], rammus_scores[a]) - auroc(rms_scores[h], rms_scores[a]))
low, high = np.percentile(differences, [2.5, 97.5])
lines.append(f"\nAUROC difference, speed-matched RAMMUS - RMS: 95% CI [{low:.2f}, {high:.2f}]")

# f: repeat with other random splits to see how much the AUROC depends on which healthy recordings were picked
lines.append(f"\nAUROC over {len(seeds)} random splits: mean [min, max]")
seed_rows = ["seed,mode,detector,fault,auroc"]
for seed in seeds:
    seed_test, seed_results = (test_clips, results) if seed == seeds[0] else run(seed)
    seed_faulty = faulty[seed_test]
    for (mode, name), (scores, _, _) in seed_results.items():
        for c in ["all faults"] + fault_classes:
            of_class = seed_faulty if c == "all faults" else clip_class[seed_test] == c
            seed_rows.append(f"{seed},{mode},{name},{c},{auroc(scores[~seed_faulty], scores[of_class]):.4f}")
(results_dir / "auroc_by_seed.csv").write_text("\n".join(seed_rows) + "\n")
rows = [r.split(",") for r in seed_rows[1:]]
for mode in ["pooled", "speed-matched"]:
    for name in ["RMS", "RAMMUS"]:
        a = [float(r[4]) for r in rows if r[1] == mode and r[2] == name and r[3] == "all faults"]
        lines.append(f"  {mode:14s} {name:7s} {np.mean(a):.2f} [{np.min(a):.2f}, {np.max(a):.2f}]")

print("\n".join(lines))
(results_dir / "results.txt").write_text("\n".join(lines) + "\n")

# g: save every test clip's scores (first split) so plots.py can draw them
with open(results_dir / "scores.csv", "w") as out:
    cutoffs = ", ".join(f"{mode} {name} {results[mode, name][2]:.4f}" for mode, name in results)
    out.write(f"# cutoffs: {cutoffs}\n")
    out.write("recording,clip,fault,speed_hz,rms_pooled,rammus_pooled,rms_matched,rammus_matched,flagged\n")
    for k, i in enumerate(test_clips):
        out.write(
            f"{names[clip_recording[i]]},{i},{clip_class[i]},{clip_speed[i]:.3f},"
            f"{results['pooled', 'RMS'][0][k]:.4f},{results['pooled', 'RAMMUS'][0][k]:.4f},"
            f"{results['speed-matched', 'RMS'][0][k]:.4f},{results['speed-matched', 'RAMMUS'][0][k]:.4f},"
            f"{feature_names[results['speed-matched', 'RAMMUS'][1][k]]}\n"
        )
