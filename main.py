from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

data_dir = Path(__file__).parent / "data" / "bearing"
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

sample_rate = 16000
frame_len = 1024
hop = 512
window = np.hanning(frame_len)
f = np.fft.rfftfreq(frame_len, d=1/sample_rate)
band_edges = [0, 500, 1000, 2000, 4000, 8001]
band_names = ["0-0.5 kHz", "0.5-1 kHz", "1-2 kHz", "2-4 kHz", "4-8 kHz"]
feature_names = [f"median {b}" for b in band_names] + [f"95th pct {b}" for b in band_names]


def load(path):
    rate, s = wavfile.read(path)
    return s.astype(float)


def speed(path):
    # section_00_source_test_anomaly_0003_vel_14.wav -> 14 (thousand rpm)
    return int(path.stem.split("_")[-1])


def rms(s):
    return np.sqrt(np.mean(s ** 2))


def band_shares(s):
    # one row per 64 ms frame, one column per band; each row adds up to 1
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


# a: find the clips
train_files = sorted((data_dir / "train").glob("section_00_source_train_normal_*.wav"))
source_test_files = sorted((data_dir / "test").glob("section_00_source_test_*.wav"))
target_test_files = sorted((data_dir / "test").glob("section_00_target_test_*.wav"))
print(f"{len(train_files)} healthy training clips")
print(f"{len(source_test_files)} test clips at training speeds")
print(f"{len(target_test_files)} test clips at new speeds")

# b: split healthy training clips 80/20 within each speed
rng = np.random.default_rng(0)
reference_files = []
calibration_files = []
for v in sorted(set(speed(p) for p in train_files)):
    clips = [p for p in train_files if speed(p) == v]
    rng.shuffle(clips)
    n_reference = int(0.8 * len(clips))
    reference_files += clips[:n_reference]
    calibration_files += clips[n_reference:]
print(f"{len(reference_files)} reference clips, {len(calibration_files)} calibration clips")

with open(results_dir / "split.txt", "w") as out:
    out.write("# seed 0\n")
    for p in reference_files:
        out.write(f"reference {p.name}\n")
    for p in calibration_files:
        out.write(f"calibration {p.name}\n")

# c: compute both detectors' measurements for every clip
def measure(files):
    rms_values = []
    features = []
    for p in files:
        s = load(p)
        rms_values.append(rms(s))
        features.append(rammus_features(s))
    return np.array(rms_values), np.array(features)

print("Computing features (takes about a minute)...")
reference_rms, reference_features = measure(reference_files)
calibration_rms, calibration_features = measure(calibration_files)
source_rms, source_features = measure(source_test_files)
target_rms, target_features = measure(target_test_files)

source_faulty = np.array(["anomaly" in p.name for p in source_test_files])
target_faulty = np.array(["anomaly" in p.name for p in target_test_files])

# d: learn what healthy looks like from the reference clips
rms_mean = np.mean(reference_rms)
rms_std = np.std(reference_rms)

feature_mean = np.mean(reference_features, axis=0)
feature_std = np.std(reference_features, axis=0)
keep = feature_std > 0


def rms_score(rms_values):
    return np.abs(rms_values - rms_mean) / rms_std


def rammus_score(features):
    z = np.abs(features[:, keep] - feature_mean[keep]) / feature_std[keep]
    return np.max(z, axis=1), np.flatnonzero(keep)[np.argmax(z, axis=1)]


# e: set each alarm cutoff at the 95th percentile of healthy calibration scores
rms_cutoff = np.percentile(rms_score(calibration_rms), 95)
rammus_cutoff = np.percentile(rammus_score(calibration_features)[0], 95)
print(f"RMS cutoff = {rms_cutoff:.2f} standard deviations")
print(f"RAMMUS cutoff = {rammus_cutoff:.2f} standard deviations")

# f: results on the test clips
lines = []
for group, rms_values, features, faulty in [
    ("training speeds", source_rms, source_features, source_faulty),
    ("new speeds", target_rms, target_features, target_faulty),
]:
    rms_scores = rms_score(rms_values)
    rammus_scores, flagged = rammus_score(features)
    lines.append(f"\nTest at {group} ({np.sum(~faulty)} healthy, {np.sum(faulty)} faulty)")
    for name, scores, cutoff in [("RMS", rms_scores, rms_cutoff), ("RAMMUS", rammus_scores, rammus_cutoff)]:
        alarms = scores > cutoff
        lines.append(
            f"  {name:7s} caught {np.sum(alarms & faulty):2d}/{np.sum(faulty)} faults, "
            f"false alarms {np.sum(alarms & ~faulty):2d}/{np.sum(~faulty)}, "
            f"AUROC {auroc(scores[~faulty], scores[faulty]):.2f}"
        )
    counts = np.bincount(flagged[faulty], minlength=len(feature_names))
    lines.append("  Measurement that gave RAMMUS its score on faulty clips:")
    for i in np.argsort(-counts):
        if counts[i] > 0:
            lines.append(f"    {feature_names[i]:20s} {counts[i]}")

# g: bootstrap 95% confidence interval on the AUROC difference at training speeds
rms_scores = rms_score(source_rms)
rammus_scores = rammus_score(source_features)[0]
healthy_idx = np.flatnonzero(~source_faulty)
faulty_idx = np.flatnonzero(source_faulty)

differences = []
for _ in range(2000):
    h = rng.choice(healthy_idx, len(healthy_idx))
    a = rng.choice(faulty_idx, len(faulty_idx))
    differences.append(auroc(rammus_scores[h], rammus_scores[a]) - auroc(rms_scores[h], rms_scores[a]))
low, high = np.percentile(differences, [2.5, 97.5])
lines.append(f"\nAUROC difference (RAMMUS - RMS) at training speeds: 95% CI [{low:.2f}, {high:.2f}]")

print("\n".join(lines))
(results_dir / "results.txt").write_text("\n".join(lines).strip() + "\n")

# h: score histograms
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name, scores, cutoff in [
    (axes[0], "RMS", rms_score(source_rms), rms_cutoff),
    (axes[1], "RAMMUS", rammus_score(source_features)[0], rammus_cutoff),
]:
    bins = np.linspace(0, max(scores.max(), cutoff) * 1.05, 30)
    ax.hist(scores[~source_faulty], bins=bins, alpha=0.6, label="Healthy")
    ax.hist(scores[source_faulty], bins=bins, alpha=0.6, label="Faulty")
    ax.axvline(cutoff, color="black", linestyle="--", label="Cutoff")
    ax.set_title(f"{name} scores at training speeds")
    ax.set_xlabel("Score (standard deviations from healthy)")
    ax.set_ylabel("Number of clips")
    ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "scores.png", dpi=150)
plt.show()

# i: spectrogram of the highest-scoring faulty clip next to a healthy clip at the same speed
scores, flagged = rammus_score(source_features)
worst = np.flatnonzero(source_faulty)[np.argmax(scores[source_faulty])]
faulty_file = source_test_files[worst]
healthy_file = [p for p in source_test_files if "normal" in p.name and speed(p) == speed(faulty_file)][0]
band = flagged[worst] % len(band_names)

spectrograms = []
for p in [healthy_file, faulty_file]:
    s = load(p)
    s = s - np.mean(s)
    frames = [s[i:i + frame_len] * window for i in range(0, len(s) - frame_len + 1, hop)]
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    spectrograms.append(10 * np.log10(P.T + 1e-12))
t = np.arange(spectrograms[0].shape[1]) * hop / sample_rate
vmin = min(np.percentile(S_db, 1) for S_db in spectrograms)
vmax = max(np.percentile(S_db, 99.9) for S_db in spectrograms)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
for ax, p, S_db, title in [(axes[0], healthy_file, spectrograms[0], "Healthy"), (axes[1], faulty_file, spectrograms[1], "Faulty")]:
    image = ax.pcolormesh(t, f, S_db, shading="auto", vmin=vmin, vmax=vmax)
    for edge in [band_edges[band], min(band_edges[band + 1], 8000)]:
        ax.axhline(edge, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{title}, {speed(p)},000 rpm")
    ax.set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Hz)")
fig.colorbar(image, ax=axes, label="Power (dB)")
fig.suptitle(f"RAMMUS flagged: {feature_names[flagged[worst]]} (between the red lines)")
plt.savefig(results_dir / "spectrogram.png", dpi=150)
plt.show()
