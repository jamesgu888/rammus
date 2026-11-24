"""
The figures: ROC curves, AUROC by fault type, the effect of speed matching,
and a spectrogram and average spectrum of a healthy and a faulty recording.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rammus import band_edges, band_names, freqs, hop, sample_rate, spectrogram

results_dir = Path(__file__).parent / "results"
colors = {"RMS": "tab:gray", "RAMMUS": "tab:blue"}
fault_labels = {
    "bearing ball_fault": "Bearing\nball",
    "bearing cage_fault": "Bearing\ncage",
    "bearing outer_race": "Bearing\nouter race",
    "horizontal-misalignment": "Misaligned\nshaft",
    "imbalance": "Imbalance",
}


def save(name):
    plt.tight_layout()
    plt.savefig(results_dir / name, dpi=150)
    plt.show()


def roc_curve(scores, faulty):
    # slide the cutoff down from the highest score and record both rates at each step
    cutoffs = np.concatenate([[np.inf], np.sort(scores)[::-1]])
    false_alarm_rate = [np.mean(scores[~faulty] >= c) for c in cutoffs]
    detection_rate = [np.mean(scores[faulty] >= c) for c in cutoffs]
    return false_alarm_rate, detection_rate


def plot_roc(curves, faulty):
    plt.figure(figsize=(5.5, 5.5))
    for label, scores, color, style in curves:
        x, y = roc_curve(scores, faulty)
        plt.plot(x, y, color=color, linestyle=style, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], color="black", linestyle=":", label="Coin flip")
    plt.xlabel("False alarm rate (healthy clips flagged)")
    plt.ylabel("Detection rate (faulty clips caught)")
    plt.title("ROC curves, all faults")
    plt.legend(loc="lower right", fontsize=9)
    save("roc.png")


def bars_with_range(groups, aurocs, labels):
    # one gray and one blue bar per group: mean AUROC over the random splits, whiskers from lowest to highest
    x = np.arange(len(groups))
    for offset, name in [(-0.2, "RMS"), (0.2, "RAMMUS")]:
        values = [aurocs[name][g] for g in groups]
        mean = np.array([np.mean(v) for v in values])
        low = mean - np.array([np.min(v) for v in values])
        high = np.array([np.max(v) for v in values]) - mean
        plt.bar(x + offset, mean, width=0.4, color=colors[name], yerr=[low, high], capsize=3, label=name)
        for xi, m, top in zip(x + offset, mean, mean + high):
            plt.text(xi, top + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    plt.axhline(0.5, color="black", linestyle=":", label="Coin flip", zorder=0)
    plt.xticks(x, labels)
    plt.ylim(0, 1.12)
    plt.ylabel("AUROC")


def plot_auroc_by_fault(aurocs, fault_types):
    plt.figure(figsize=(9, 4.5))
    bars_with_range(fault_types, aurocs["speed-matched"], [fault_labels[f] for f in fault_types])
    plt.title("AUROC by fault type (speed-matched, mean of 5 random splits)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save("auroc_by_fault.png")


def plot_speed_matching(aurocs):
    plt.figure(figsize=(6, 4.5))
    both = {name: {mode: aurocs[mode][name]["all faults"] for mode in aurocs} for name in ["RMS", "RAMMUS"]}
    bars_with_range(["pooled", "speed-matched"], both,
                    ["Compared with all\nhealthy recordings", "Compared with healthy\nrecordings at similar speed"])
    plt.title("Comparing at the same speed helps both detectors")
    plt.legend(loc="upper left")
    save("speed_matching.png")


def plot_spectrogram(healthy, faulty, titles, band):
    spectrograms = [10 * np.log10(spectrogram(s[:sample_rate]).T + 1e-12) for s in [healthy, faulty]]
    t = np.arange(spectrograms[0].shape[1]) * hop / sample_rate
    vmin = min(np.percentile(S, 1) for S in spectrograms)
    vmax = max(np.percentile(S, 99.9) for S in spectrograms)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, S, title in zip(axes, spectrograms, titles):
        image = ax.pcolormesh(t, freqs / 1000, S, shading="auto", vmin=vmin, vmax=vmax)
        for edge in band_edges[band:band + 2]:
            ax.axhline(min(edge, 12500) / 1000, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (kHz)")
    fig.colorbar(image, ax=axes, label="Power (dB)")
    fig.suptitle(f"RAMMUS flagged the {band_names[band]} band (between the red lines)")
    plt.savefig(results_dir / "spectrogram.png", dpi=150)
    plt.show()


def plot_spectrum(healthy, faulty, titles, band):
    plt.figure(figsize=(9, 4))
    for s, title, color in [(healthy, titles[0], "tab:green"), (faulty, titles[1], "tab:red")]:
        P = np.mean(spectrogram(s), axis=0)
        plt.plot(freqs / 1000, 10 * np.log10(P / np.sum(P)), color=color, linewidth=1, label=title)
    plt.axvspan(band_edges[band] / 1000, min(band_edges[band + 1], 12500) / 1000,
                color="tab:blue", alpha=0.1, label=f"Flagged band ({band_names[band]})")
    for edge in band_edges[1:-1]:
        plt.axvline(edge / 1000, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Share of total power (dB)")
    plt.title("Average spectrum (dotted lines: RAMMUS band edges)")
    plt.legend(loc="upper right")
    save("spectrum.png")
