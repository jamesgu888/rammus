from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Figures for the write-up. Run main.py first; it writes results/scores.csv and results/auroc_by_seed.csv.
data_dir = Path(__file__).parent / "data"
results_dir = Path(__file__).parent / "results"

rms_color = "tab:gray"
rammus_color = "tab:blue"
fault_labels = {
    "bearing ball_fault": "Bearing\nball",
    "bearing cage_fault": "Bearing\ncage",
    "bearing outer_race": "Bearing\nouter race",
    "horizontal-misalignment": "Misaligned\nshaft",
    "imbalance": "Imbalance",
}

# Same FFT settings as main.py, for the spectrogram
sample_rate = 25000
frame_len = 1024
hop = 512
window = np.hanning(frame_len)
f = np.fft.rfftfreq(frame_len, d=1/sample_rate)
band_edges = [0, 781.25, 1562.5, 3125, 6250, 12500]


def auroc(healthy_scores, faulty_scores):
    higher = faulty_scores[:, None] > healthy_scores[None, :]
    tied = faulty_scores[:, None] == healthy_scores[None, :]
    return np.mean(higher) + 0.5 * np.mean(tied)


def roc_curve(scores, faulty):
    # slide the cutoff from above the highest score down to the lowest, recording the two rates at each step
    cutoffs = np.concatenate([[np.inf], np.sort(scores)[::-1]])
    false_alarm_rate = [np.mean(scores[~faulty] >= c) for c in cutoffs]
    detection_rate = [np.mean(scores[faulty] >= c) for c in cutoffs]
    return false_alarm_rate, detection_rate


lines = (results_dir / "scores.csv").read_text().splitlines()
cutoff_parts = lines[0].replace("# cutoffs: ", "").split(", ")
cutoffs = {" ".join(p.split()[:2]): float(p.split()[2]) for p in cutoff_parts}
rows = [line.split(",") for line in lines[2:]]
recording = np.array([r[0] for r in rows])
fault = np.array([r[2] for r in rows])
speed = np.array([float(r[3]) for r in rows])
scores = {
    ("pooled", "RMS"): np.array([float(r[4]) for r in rows]),
    ("pooled", "RAMMUS"): np.array([float(r[5]) for r in rows]),
    ("speed-matched", "RMS"): np.array([float(r[6]) for r in rows]),
    ("speed-matched", "RAMMUS"): np.array([float(r[7]) for r in rows]),
}
flagged = np.array([r[8] for r in rows])
faulty = fault != "normal"
fault_classes = list(fault_labels)

seed_rows = [line.split(",") for line in (results_dir / "auroc_by_seed.csv").read_text().splitlines()[1:]]


def seed_aurocs(mode, name, c):
    return np.array([float(r[4]) for r in seed_rows if r[1] == mode and r[2] == name and r[3] == c])


# a: ROC curves
plt.figure(figsize=(5.5, 5.5))
for mode, name, color, style in [
    ("speed-matched", "RAMMUS", rammus_color, "-"),
    ("speed-matched", "RMS", rms_color, "-"),
    ("pooled", "RAMMUS", rammus_color, "--"),
]:
    s = scores[mode, name]
    x, y = roc_curve(s, faulty)
    plt.plot(x, y, color=color, linestyle=style, linewidth=2,
             label=f"{name}, {mode} (AUROC {auroc(s[~faulty], s[faulty]):.2f})")
plt.plot([0, 1], [0, 1], color="black", linestyle=":", label="Coin flip (AUROC 0.50)")
plt.xlabel("False alarm rate (healthy clips flagged)")
plt.ylabel("Detection rate (faulty clips caught)")
plt.title("ROC curves, all faults")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(results_dir / "roc.png", dpi=150)
plt.show()

# b: AUROC for each fault type, speed-matched, mean over the 5 random splits (bars show the lowest and highest)
x = np.arange(len(fault_classes))
plt.figure(figsize=(9, 4.5))
for offset, name, color in [(-0.2, "RMS", rms_color), (0.2, "RAMMUS", rammus_color)]:
    a = [seed_aurocs("speed-matched", name, c) for c in fault_classes]
    mean = np.array([np.mean(v) for v in a])
    error = np.array([[m - np.min(v) for m, v in zip(mean, a)], [np.max(v) - m for m, v in zip(mean, a)]])
    plt.bar(x + offset, mean, width=0.4, color=color, yerr=error, capsize=3, label=name)
    for xi, m, top in zip(x + offset, mean, mean + error[1]):
        plt.text(xi, top + 0.02, f"{m:.2f}", ha="center", fontsize=9)
plt.axhline(0.5, color="black", linestyle=":", label="Coin flip", zorder=0)
plt.xticks(x, [fault_labels[c] for c in fault_classes])
plt.ylim(0, 1.12)
plt.ylabel("AUROC")
plt.title("AUROC by fault type (speed-matched, mean of 5 random splits)")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(results_dir / "auroc_by_fault.png", dpi=150)
plt.show()

# c: why speed matching matters: overall AUROC with and without it
x = np.arange(2)
plt.figure(figsize=(6, 4.5))
for offset, name, color in [(-0.2, "RMS", rms_color), (0.2, "RAMMUS", rammus_color)]:
    a = [seed_aurocs(mode, name, "all faults") for mode in ["pooled", "speed-matched"]]
    mean = np.array([np.mean(v) for v in a])
    error = np.array([[m - np.min(v) for m, v in zip(mean, a)], [np.max(v) - m for m, v in zip(mean, a)]])
    plt.bar(x + offset, mean, width=0.4, color=color, yerr=error, capsize=3, label=name)
    for xi, m, top in zip(x + offset, mean, mean + error[1]):
        plt.text(xi, top + 0.02, f"{m:.2f}", ha="center")
plt.axhline(0.5, color="black", linestyle=":", label="Coin flip", zorder=0)
plt.xticks(x, ["Compared with all\nhealthy recordings", "Compared with healthy\nrecordings at similar speed"])
plt.ylim(0, 1.12)
plt.ylabel("AUROC, all faults")
plt.title("Comparing at the same speed helps both detectors")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(results_dir / "speed_matching.png", dpi=150)
plt.show()

# d: score histograms, speed-matched. Scores span 0.1 to thousands, so the x axis is logarithmic,
# and there are about 100 faulty clips per healthy one, so each group's bars show its fraction of clips.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
bins = np.geomspace(0.1, max(scores["speed-matched", n].max() for n in ["RMS", "RAMMUS"]) * 1.1, 40)
for ax, name in zip(axes, ["RMS", "RAMMUS"]):
    s = np.clip(scores["speed-matched", name], 0.1, None)
    for label, group in [("Healthy", ~faulty), ("Faulty", faulty)]:
        ax.hist(s[group], bins=bins, weights=np.ones(np.sum(group)) / np.sum(group), alpha=0.6, label=label)
    ax.axvline(cutoffs["speed-matched " + name], color="black", linestyle="--", label="Cutoff")
    ax.set_xscale("log")
    ax.set_title(f"{name} scores (speed-matched)")
    ax.set_xlabel("Score (standard deviations from healthy)")
    ax.set_ylabel("Fraction of clips")
    ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(results_dir / "scores.png", dpi=150)
plt.show()

# e: spectrogram of a healthy recording next to a bearing outer race fault at a similar shaft speed
saved = np.load(data_dir / "mafaulda_microphone_25k.npz")
names = list(saved["names"])
healthy_name = sorted(set(recording[fault == "normal"]), key=lambda n: abs(float(Path(n).stem) - 40))[0]
fault_candidates = sorted(set(recording[fault == "bearing outer_race"]),
                          key=lambda n: abs(float(Path(n).stem) - float(Path(healthy_name).stem)))
faulty_name = fault_candidates[0]
band = flagged[recording == faulty_name][0]
band_names = ["0-0.8 kHz", "0.8-1.6 kHz", "1.6-3.1 kHz", "3.1-6.3 kHz", "6.3-12.5 kHz"]
band_index = next(i for i, b in enumerate(band_names) if band.endswith(b))
flagged_edges = band_edges[band_index:band_index + 2]

spectrograms = []
for name in [healthy_name, faulty_name]:
    s = saved["signals"][names.index(name)][:sample_rate].astype(float)  # first 1 s clip
    s = s - np.mean(s)
    frames = [s[i:i + frame_len] * window for i in range(0, len(s) - frame_len + 1, hop)]
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    spectrograms.append(10 * np.log10(P.T + 1e-12))
t = np.arange(spectrograms[0].shape[1]) * hop / sample_rate
vmin = min(np.percentile(S, 1) for S in spectrograms)
vmax = max(np.percentile(S, 99.9) for S in spectrograms)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
for ax, name, S, title in [(axes[0], healthy_name, spectrograms[0], "Healthy"),
                           (axes[1], faulty_name, spectrograms[1], "Bearing outer race fault")]:
    image = ax.pcolormesh(t, f / 1000, S, shading="auto", vmin=vmin, vmax=vmax)
    for edge in flagged_edges:
        ax.axhline(edge / 1000, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{title}, shaft at {float(Path(name).stem):.1f} Hz")
    ax.set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (kHz)")
fig.colorbar(image, ax=axes, label="Power (dB)")
fig.suptitle(f"RAMMUS flagged this clip on: {band} (between the red lines)")
plt.savefig(results_dir / "spectrogram.png", dpi=150)
plt.show()

# f: average spectrum of the same two recordings (all 5 s), with the flagged band shaded
plt.figure(figsize=(9, 4))
for name, label, color in [(healthy_name, "Healthy", "tab:green"), (faulty_name, "Bearing outer race fault", "tab:red")]:
    s = saved["signals"][names.index(name)].astype(float)
    s = s - np.mean(s)
    frames = [s[i:i + frame_len] * window for i in range(0, len(s) - frame_len + 1, hop)]
    P = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    plt.plot(f / 1000, 10 * np.log10(P / np.sum(P)), color=color, linewidth=1, label=label)
plt.axvspan(flagged_edges[0] / 1000, flagged_edges[1] / 1000, color="tab:blue", alpha=0.1, label=f"Flagged band ({band_names[band_index]})")
for edge in band_edges[1:-1]:
    plt.axvline(edge / 1000, color="gray", linestyle=":", linewidth=1)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Share of total power (dB)")
plt.title(f"Average spectrum, shaft at about {float(Path(healthy_name).stem):.0f} Hz (dotted lines: RAMMUS band edges)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(results_dir / "spectrum.png", dpi=150)
plt.show()
