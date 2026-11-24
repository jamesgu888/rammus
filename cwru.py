from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

data_dir = Path(__file__).parent / "data" / "cwru"
results_dir = Path(__file__).parent / "results" / "cwru"
data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Case Western Reserve University Bearing Data Center, engineering.case.edu/bearingdatacenter
# File numbers copied from the site's "Normal Baseline Data" and "48k Drive End Bearing Fault Data" tables.
# The normal baseline is only recorded at 48 kHz, so we use the 48 kHz drive-end fault files too.
# That way every recording has the same sampling rate and nothing has to be resampled.
sample_rate = 48000
recordings = []  # (file number, condition, fault diameter in inches, motor load in hp)
for hp, number in enumerate([97, 98, 99, 100]):
    recordings.append((number, "healthy", 0, hp))
for condition, size, numbers in [
    ("inner race", 0.007, [109, 110, 111, 112]),  # one file per motor load: 0, 1, 2, 3 hp
    ("ball", 0.007, [122, 123, 124, 125]),
    ("outer race", 0.007, [135, 136, 137, 138]),  # outer race fault centered at 6 o'clock
    ("inner race", 0.014, [174, 175, 176, 177]),
    ("ball", 0.014, [189, 190, 191, 192]),
    ("outer race", 0.014, [201, 202, 203, 204]),
    ("inner race", 0.021, [213, 214, 215, 217]),  # the site has no 216; 3 hp is 217
    ("ball", 0.021, [226, 227, 228, 229]),
    ("outer race", 0.021, [238, 239, 240, 241]),
]:
    for hp, number in enumerate(numbers):
        recordings.append((number, condition, size, hp))
fault_types = [("inner race", 0.007), ("ball", 0.007), ("outer race", 0.007),
               ("inner race", 0.014), ("ball", 0.014), ("outer race", 0.014),
               ("inner race", 0.021), ("ball", 0.021), ("outer race", 0.021)]

# There are only about 35 s of healthy recordings in total, so we cut 0.25 s clips to get enough of them.
clip_len = sample_rate // 4
# Same 1024-sample frames and 512 hop as main.py, so the code is identical.
# At 48 kHz that is a 21 ms frame (main.py: 64 ms at 16 kHz), and about 22 frames per clip.
frame_len = 1024
hop = 512
window = np.hanning(frame_len)
f = np.fft.rfftfreq(frame_len, d=1/sample_rate)
# Octave bands up to the 24 kHz Nyquist frequency, the same pattern as main.py's 0-0.5/0.5-1/1-2/2-4/4-8 kHz at 16 kHz
band_edges = [0, 1500, 3000, 6000, 12000, 24001]
band_names = ["0-1.5 kHz", "1.5-3 kHz", "3-6 kHz", "6-12 kHz", "12-24 kHz"]
feature_names = [f"median {b}" for b in band_names] + [f"95th pct {b}" for b in band_names]


def load(number):
    # drive-end accelerometer signal and the rpm stored in the file (nan if the file has none)
    m = loadmat(data_dir / f"{number}.mat")
    key = f"X{number:03d}_DE_time"
    if key not in m:
        key = [k for k in m if k.endswith("_DE_time")][0]  # 174.mat only holds a 1.3 s record named X173
    # 99.mat and 217.mat also hold a copy of another record (X098, X215); the key above picks the right one.
    # 98.mat, 99.mat and 174.mat store no rpm, so section a also estimates it from the spectrum.
    rpm = m.get(f"X{number:03d}RPM", [[np.nan]])[0][0]
    return m[key].ravel(), rpm


def shaft_rpm(s):
    # the biggest spectral peak between 27 and 32 Hz (1620-1920 rpm) is the shaft turning once per revolution
    S = np.abs(np.fft.rfft(s - np.mean(s)))
    freqs = np.fft.rfftfreq(len(s), d=1/sample_rate)
    near = (freqs > 27) & (freqs < 32)
    return 60 * freqs[near][np.argmax(S[near])]


def rms(s):
    return np.sqrt(np.mean(s ** 2))


def band_shares(s):
    # one row per 21 ms frame, one column per band; each row adds up to 1
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
# so that section f can refit them on different reference clips.
def rms_score(reference_rms, rms_values):
    return np.abs(rms_values - np.mean(reference_rms)) / np.std(reference_rms)


def rammus_score(reference_features, features):
    feature_mean = np.mean(reference_features, axis=0)
    feature_std = np.std(reference_features, axis=0)
    keep = feature_std > 0
    z = np.abs(features[:, keep] - feature_mean[keep]) / feature_std[keep]
    return np.max(z, axis=1), np.flatnonzero(keep)[np.argmax(z, axis=1)]


# a: download the recordings, check them, and cut them into clips
clip_rms = []
clip_features = []
clip_condition = []
clip_size = []
clip_hp = []
print("file  condition   size    load  samples (seconds)  stored rpm  rpm from spectrum")
for number, condition, size, hp in recordings:
    path = data_dir / f"{number}.mat"
    if not path.exists():
        urlretrieve(f"https://engineering.case.edu/sites/default/files/{number}.mat", path)
    s, rpm = load(number)
    print(f"{number:4d}  {condition:10s}  {size:.3f}\"  {hp} hp  {len(s):6d} ({len(s) / sample_rate:4.1f} s)"
          f"     {rpm:6.0f}      {shaft_rpm(s):6.0f}")
    for start in range(0, len(s) - clip_len + 1, clip_len):
        clip = s[start:start + clip_len]
        clip_rms.append(rms(clip))
        clip_features.append(rammus_features(clip))
        clip_condition.append(condition)
        clip_size.append(size)
        clip_hp.append(hp)
clip_rms = np.array(clip_rms)
clip_features = np.array(clip_features)
clip_condition = np.array(clip_condition)
clip_size = np.array(clip_size)
clip_hp = np.array(clip_hp)
faulty = clip_condition != "healthy"
print(f"{np.sum(~faulty)} healthy clips, {np.sum(faulty)} faulty clips, {clip_len / sample_rate} s each")


# b: split healthy clips 60/20/20 into reference / calibration / test within each motor load
rng = np.random.default_rng(0)
group = np.full(len(faulty), "test", dtype=object)  # every faulty clip is a test clip
for hp in range(4):
    clips = np.flatnonzero(~faulty & (clip_hp == hp))
    rng.shuffle(clips)
    n_reference = int(0.6 * len(clips))
    n_calibration = int(0.2 * len(clips))
    group[clips[:n_reference]] = "reference"
    group[clips[n_reference:n_reference + n_calibration]] = "calibration"
print(f"{np.sum(group == 'reference')} reference clips, {np.sum(group == 'calibration')} calibration clips, "
      f"{np.sum(~faulty & (group == 'test'))} healthy test clips")


# c: fit both detectors on the reference clips, put each cutoff at the 95th percentile of the
# calibration clips' scores, then count alarms on the test clips (all three are true/false masks over clips)
def evaluate(title, reference, calibration, test, details):
    lines = [f"\n{title} ({np.sum(test & ~faulty)} healthy, {np.sum(test & faulty)} faulty)"]
    is_faulty = faulty[test]
    results = {}
    for name in ["RMS", "RAMMUS"]:
        if name == "RMS":
            calibration_scores = rms_score(clip_rms[reference], clip_rms[calibration])
            scores = rms_score(clip_rms[reference], clip_rms[test])
        else:
            calibration_scores = rammus_score(clip_features[reference], clip_features[calibration])[0]
            scores = rammus_score(clip_features[reference], clip_features[test])[0]
        cutoff = np.percentile(calibration_scores, 95)
        alarms = scores > cutoff
        caught = np.sum(alarms & is_faulty)
        passed = np.sum(~alarms & ~is_faulty)
        # plain accuracy is dominated by the many faulty clips, so also give the balanced accuracy:
        # the average of the catch rate and the pass rate, i.e. the accuracy if both groups were the same size
        accuracy = (caught + passed) / len(scores)
        balanced = (caught / np.sum(is_faulty) + passed / np.sum(~is_faulty)) / 2
        lines.append(
            f"  {name:7s} caught {caught:4d}/{np.sum(is_faulty)} faults, "
            f"false alarms {np.sum(alarms & ~is_faulty):2d}/{np.sum(~is_faulty)}, "
            f"accuracy {accuracy:.3f} (balanced {balanced:.3f}), "
            f"AUROC {auroc(scores[~is_faulty], scores[is_faulty]):.3f}, cutoff {cutoff:.1f}"
        )
        results[name] = (scores, alarms, cutoff)
    if details:
        lines.append("  Faults caught by type:  RMS        RAMMUS")
        for condition, size in fault_types:
            of_type = (clip_condition[test] == condition) & (clip_size[test] == size)
            rms_caught = f"{np.sum(results['RMS'][1] & of_type)}/{np.sum(of_type)}"
            rammus_caught = f"{np.sum(results['RAMMUS'][1] & of_type)}/{np.sum(of_type)}"
            lines.append(f"    {condition:10s} {size:.3f}\"     {rms_caught:10s} {rammus_caught}")
        flagged = rammus_score(clip_features[reference], clip_features[test])[1]
        counts = np.bincount(flagged[is_faulty], minlength=len(feature_names))
        lines.append("  Measurement that gave RAMMUS its score on faulty clips:")
        for i in np.argsort(-counts):
            if counts[i] > 0:
                lines.append(f"    {feature_names[i]:20s} {counts[i]}")
    return lines, results


test = group == "test"
lines, results = evaluate("Test with all 4 loads", group == "reference", group == "calibration", test, True)
lines.append("\n  Caveat: the healthy test clips are cut from the same 4 recordings as the reference clips,")
lines.append("  so they look more like the reference than a new healthy recording would.")

# d: leave one load out. Healthy clips from the other 3 loads train the detectors
# (section b's reference + calibration clips become the reference, its healthy test clips the calibration),
# and the test is every clip at the held-out load, healthy and faulty.
lines.append("\nLeave one load out (learn healthy from the other 3 loads, test on this one)")
for hp in range(4):
    reference = ~faulty & (clip_hp != hp) & (group != "test")
    calibration = ~faulty & (clip_hp != hp) & (group == "test")
    fold_lines, _ = evaluate(f"Test at {hp} hp", reference, calibration, clip_hp == hp, False)
    lines += fold_lines

print("\n".join(lines))
(results_dir / "results.txt").write_text("\n".join(lines).strip() + "\n")

# e: score histograms for the test with all 4 loads
# Faulty scores reach the hundreds, so the x axis is logarithmic (scores below 0.1 go in the first bin).
# There are 44 times more faulty clips than healthy ones, so each group's bars show its fraction of clips.
# Both panels share one x axis; the RAMMUS scores above 100 are 2 clips with a burst of 12-24 kHz noise.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
max_score = max(results["RMS"][0].max(), results["RAMMUS"][0].max())
for ax, name in [(axes[0], "RMS"), (axes[1], "RAMMUS")]:
    scores, alarms, cutoff = results[name]
    healthy_scores = np.clip(scores[~faulty[test]], 0.1, None)
    faulty_scores = scores[faulty[test]]
    bins = np.geomspace(0.1, max_score * 1.1, 40)
    ax.hist(healthy_scores, bins=bins, weights=np.ones(len(healthy_scores)) / len(healthy_scores), alpha=0.6, label="Healthy")
    ax.hist(faulty_scores, bins=bins, weights=np.ones(len(faulty_scores)) / len(faulty_scores), alpha=0.6, label="Faulty")
    ax.axvline(cutoff, color="black", linestyle="--", label="Cutoff")
    ax.set_xscale("log")
    ax.set_title(f"{name} scores on CWRU test clips")
    ax.set_xlabel("Score (standard deviations from healthy)")
    ax.set_ylabel("Fraction of clips")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # room for the legend above the bars
    ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(results_dir / "scores.png", dpi=150)
plt.show()
