from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Summary figures for the write-up. Run main.py first (for each machine) so results/<machine>/scores.csv exists.
results_dir = Path(__file__).parent / "results"
out_dir = results_dir / "summary"
out_dir.mkdir(exist_ok=True)

rms_color = "tab:gray"
rammus_color = "tab:blue"


def load_scores(machine):
    lines = (results_dir / machine / "scores.csv").read_text().splitlines()
    cutoffs = lines[0].replace(",", "").split()
    rows = [line.split(",") for line in lines[2:]]
    clips = {
        "group": np.array([r[1] for r in rows]),
        "setting": np.array([r[2] for r in rows]),
        "faulty": np.array([r[3] == "1" for r in rows]),
        "rms": np.array([float(r[4]) for r in rows]),
        "rammus": np.array([float(r[5]) for r in rows]),
        "flagged": np.array([r[6] for r in rows]),
    }
    return clips, float(cutoffs[2]), float(cutoffs[4])


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


clips, rms_cutoff, rammus_cutoff = load_scores("bearing")
training = clips["group"] == "training"
new = clips["group"] == "new"
speed = np.array([int(s.split("_")[1]) for s in clips["setting"]])

# a: ROC curves at training speeds
faulty = clips["faulty"][training]
plt.figure(figsize=(5, 5))
for name, key, color in [("RMS", "rms", rms_color), ("RAMMUS", "rammus", rammus_color)]:
    scores = clips[key][training]
    x, y = roc_curve(scores, faulty)
    plt.plot(x, y, color=color, linewidth=2, label=f"{name} (AUROC {auroc(scores[~faulty], scores[faulty]):.2f})")
plt.plot([0, 1], [0, 1], color="black", linestyle=":", label="Coin flip (AUROC 0.50)")
plt.xlabel("False alarm rate (healthy clips flagged)")
plt.ylabel("Detection rate (faulty clips caught)")
plt.title("Bearing, training speeds")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(out_dir / "roc.png", dpi=150)
plt.show()

# b: caught, false alarms and accuracy, at training speeds and at new speeds
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
x = np.arange(2)
for ax, title in zip(axes, ["Faults caught", "False alarms", "Accuracy"]):
    for offset, name, key, cutoff, color in [(-0.2, "RMS", "rms", rms_cutoff, rms_color), (0.2, "RAMMUS", "rammus", rammus_cutoff, rammus_color)]:
        values = []
        for group in [training, new]:
            alarms = clips[key][group] > cutoff
            faulty = clips["faulty"][group]
            if title == "Faults caught":
                values.append(np.mean(alarms[faulty]))
            elif title == "False alarms":
                values.append(np.mean(alarms[~faulty]))
            else:
                values.append(np.mean(alarms == faulty))
        bars = ax.bar(x + offset, np.array(values) * 100, width=0.4, color=color, label=name)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xticks(x, ["Training speeds", "New speeds"])
    ax.set_title(title)
    ax.set_ylim(0, 105)
axes[0].set_ylabel("Percent of clips")
axes[2].axhline(50, color="black", linestyle=":", label="Coin flip", zorder=0)
axes[2].legend(loc="upper right")
fig.suptitle("Bearing: how often each detector is right")
plt.tight_layout()
plt.savefig(out_dir / "rates.png", dpi=150)
plt.show()

# c: false alarm rate on healthy clips at each shaft speed
speeds = np.unique(speed)
plt.figure(figsize=(9, 4))
plt.axvspan(6, 22, color="tab:green", alpha=0.1, label="Training speed range")
for name, key, cutoff, color in [("RMS", "rms", rms_cutoff, rms_color), ("RAMMUS", "rammus", rammus_cutoff, rammus_color)]:
    rates = [np.mean(clips[key][(speed == v) & ~clips["faulty"]] > cutoff) * 100 for v in speeds]
    plt.plot(speeds, rates, marker="o", color=color, linewidth=2, label=name)
plt.axhline(5, color="black", linestyle=":", label="Target (5%)")
training_speeds = np.unique(speed[training])
plt.scatter(training_speeds, np.full(len(training_speeds), -6), marker="^", color="tab:green", label="Speeds it trained on", clip_on=False)
plt.xticks(speeds)
plt.ylim(-8, 105)
plt.xlabel("Shaft speed (thousand rpm)")
plt.ylabel("Healthy clips flagged (%)")
plt.title("Bearing: false alarms by speed (6 to 10 healthy clips per speed)")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(out_dir / "false_alarms_by_speed.png", dpi=150)
plt.show()

# d: which measurement set off RAMMUS on the faulty clips it caught at training speeds
caught = training & clips["faulty"] & (clips["rammus"] > rammus_cutoff)
names, counts = np.unique(clips["flagged"][caught], return_counts=True)
order = np.argsort(counts)
plt.figure(figsize=(7, 4))
plt.barh(names[order], counts[order], color=rammus_color)
plt.xlabel("Number of caught faults")
plt.title("Bearing: which band set off RAMMUS")
plt.tight_layout()
plt.savefig(out_dir / "flagged_bands.png", dpi=150)
plt.show()

# e: AUROC on every machine, at training settings
machines = ["bearing", "gearbox", "slider", "valve", "fan"]
rms_auroc = []
rammus_auroc = []
for m in machines:
    c = load_scores(m)[0]
    t = c["group"] == "training"
    f = c["faulty"][t]
    rms_auroc.append(auroc(c["rms"][t][~f], c["rms"][t][f]))
    rammus_auroc.append(auroc(c["rammus"][t][~f], c["rammus"][t][f]))

x = np.arange(len(machines))
plt.figure(figsize=(8, 4))
bars = plt.bar(x - 0.2, rms_auroc, width=0.4, color=rms_color, label="RMS")
plt.bar_label(bars, fmt="%.2f", padding=2)
bars = plt.bar(x + 0.2, rammus_auroc, width=0.4, color=rammus_color, label="RAMMUS")
plt.bar_label(bars, fmt="%.2f", padding=2)
plt.axhline(0.5, color="black", linestyle=":", label="Coin flip")
plt.xticks(x, ["Bearing", "Gearbox", "Slider", "Valve", "Fan*"])
plt.ylim(0, 1.1)
plt.ylabel("AUROC")
plt.title("Every MIMII DG machine, training settings")
plt.figtext(0.01, 0.01, "*Fan's healthy test clips are quieter than all its training clips, so its numbers are not trustworthy.", fontsize=8)
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(out_dir / "machines.png", dpi=150)
plt.show()
