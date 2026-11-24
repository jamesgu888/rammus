from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# DCASE 2020 Task 2 development data (Zenodo record 3678171): six machine types, 16 kHz, about 10 s per clip.
# Unlike MIMII DG, the test clips are recorded under the same conditions as the training clips,
# so there is no "new speed" problem. Each machine type has several individual machines (ids),
# and each id gets its own healthy reference, the way the competition scored it.
data_dir = Path(__file__).parent / "data" / "dcase2020"
results_dir = Path(__file__).parent / "results" / "dcase2020"
results_dir.mkdir(parents=True, exist_ok=True)

# AUROC of the official autoencoder baseline on this data, averaged over ids (DCASE 2020 Task 2 page)
baseline_auroc = {"ToyCar": 0.788, "ToyConveyor": 0.725, "fan": 0.658, "pump": 0.729, "slider": 0.848, "valve": 0.663}

sample_rate = 16000
frame_len = 1024
hop = 512
window = np.hanning(frame_len)
f = np.fft.rfftfreq(frame_len, d=1/sample_rate)
band_edges = [0, 500, 1000, 2000, 4000, 8001]


def load(path):
    rate, s = wavfile.read(path)
    if s.ndim > 1:
        s = s[:, 0]
    return s.astype(float)


def machine_id(path):
    # normal_id_02_00000012.wav -> "id_02"
    return "_".join(path.stem.split("_")[1:3])


def rms(s):
    return np.sqrt(np.mean(s ** 2))


def band_shares(s):
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
    higher = faulty_scores[:, None] > healthy_scores[None, :]
    tied = faulty_scores[:, None] == healthy_scores[None, :]
    return np.mean(higher) + 0.5 * np.mean(tied)


def measure(files):
    rms_values = []
    features = []
    for p in files:
        s = load(p)
        rms_values.append(rms(s))
        features.append(rammus_features(s))
    return np.array(rms_values)[:, None], np.array(features)


def z_scores(values, reference):
    # values and reference have one row per clip and one column per measurement (RMS has 1 column, RAMMUS 10).
    # Count how many standard deviations each measurement is from the healthy reference; the biggest is the score.
    mean = np.mean(reference, axis=0)
    std = np.std(reference, axis=0)
    keep = std > 0
    z = np.abs(values[:, keep] - mean[keep]) / std[keep]
    return np.max(z, axis=1)


rng = np.random.default_rng(0)
lines = []
mean_auroc = {"RMS": {}, "RAMMUS": {}}
for machine in sorted(p.name for p in data_dir.iterdir() if (p / "test").is_dir()):
    train_files = sorted((data_dir / machine / "train").glob("normal_*.wav"))
    test_files = sorted((data_dir / machine / "test").glob("*.wav"))
    ids = sorted(set(machine_id(p) for p in test_files))
    print(f"{machine}: {len(train_files)} training clips, {len(test_files)} test clips, {len(ids)} machines")

    results = {"RMS": [], "RAMMUS": []}
    for i in ids:
        # a: split this machine's healthy training clips 80/20 into reference and calibration
        clips = [p for p in train_files if machine_id(p) == i]
        rng.shuffle(clips)
        n_reference = int(0.8 * len(clips))
        reference_rms, reference_features = measure(clips[:n_reference])
        calibration_rms, calibration_features = measure(clips[n_reference:])

        # b: score the test clips against the reference, with the cutoff at the 95th percentile of calibration scores
        test = [p for p in test_files if machine_id(p) == i]
        test_rms, test_features = measure(test)
        faulty = np.array(["anomaly" in p.name for p in test])
        for name, test_values, reference, calibration in [
            ("RMS", test_rms, reference_rms, calibration_rms),
            ("RAMMUS", test_features, reference_features, calibration_features),
        ]:
            scores = z_scores(test_values, reference)
            cutoff = np.percentile(z_scores(calibration, reference), 95)
            alarms = scores > cutoff
            results[name].append((np.sum(alarms & faulty), np.sum(faulty), np.sum(alarms & ~faulty), np.sum(~faulty),
                                  np.mean(alarms == faulty), auroc(scores[~faulty], scores[faulty])))

    # c: add up the results over this machine type's ids
    lines.append(f"\n{machine} ({len(ids)} machines, official autoencoder baseline AUROC {baseline_auroc.get(machine, float('nan')):.2f})")
    for name in ["RMS", "RAMMUS"]:
        r = np.array(results[name])
        mean_auroc[name][machine] = np.mean(r[:, 5])
        lines.append(
            f"  {name:7s} caught {int(r[:, 0].sum()):4d}/{int(r[:, 1].sum())} faults, "
            f"false alarms {int(r[:, 2].sum()):3d}/{int(r[:, 3].sum())}, "
            f"accuracy {np.mean(r[:, 4]):.0%}, "
            f"AUROC {np.mean(r[:, 5]):.2f} (per machine: {', '.join(f'{a:.2f}' for a in r[:, 5])})"
        )

print("\n".join(lines))
(results_dir / "results.txt").write_text("\n".join(lines).strip() + "\n")

# d: AUROC for every machine type next to the official baseline
machines = list(mean_auroc["RMS"])
rms_auroc = [mean_auroc["RMS"][m] for m in machines]
rammus_auroc = [mean_auroc["RAMMUS"][m] for m in machines]

x = np.arange(len(machines))
plt.figure(figsize=(9, 4))
for offset, values, label, color in [
    (-0.27, rms_auroc, "RMS", "tab:gray"),
    (0, rammus_auroc, "RAMMUS", "tab:blue"),
    (0.27, [baseline_auroc[m] for m in machines], "Official autoencoder", "tab:orange"),
]:
    bars = plt.bar(x + offset, values, width=0.27, color=color, label=label)
    plt.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
plt.axhline(0.5, color="black", linestyle=":", label="Coin flip", zorder=0)
plt.xticks(x, machines)
plt.ylim(0, 1.1)
plt.ylabel("AUROC (average over machines)")
plt.title("DCASE 2020 machines")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(results_dir / "machines.png", dpi=150)
plt.show()
