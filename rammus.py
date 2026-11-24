import numpy as np

# RAMMUS: short FFTs, the share of power in five frequency bands, summarized over each clip.
# The microphone audio is downsampled to 25 kHz, so the highest frequency we can see is 12.5 kHz.
sample_rate = 25000
frame_len = 1024  # 41 ms per FFT
hop = 512
window = np.hanning(frame_len)
freqs = np.fft.rfftfreq(frame_len, d=1/sample_rate)

band_edges = [0, 781.25, 1562.5, 3125, 6250, 12501]
band_names = ["0-0.8 kHz", "0.8-1.6 kHz", "1.6-3.1 kHz", "3.1-6.3 kHz", "6.3-12.5 kHz"]
feature_names = [f"median {b}" for b in band_names] + [f"95th pct {b}" for b in band_names]


def spectrogram(s):
    # power of each frequency bin, one row per 41 ms frame
    s = s - np.mean(s)
    frames = [s[i:i + frame_len] * window for i in range(0, len(s) - frame_len + 1, hop)]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2


def band_shares(s):
    # share of each frame's power in each band (each row adds up to 1)
    P = spectrogram(s)
    shares = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        in_band = (freqs >= lo) & (freqs < hi)
        shares.append(np.sum(P[:, in_band], axis=1) / np.sum(P, axis=1))
    return np.array(shares).T


def rammus_features(s):
    # median and 95th percentile of each band's share over the clip: 10 numbers
    shares = band_shares(s)
    return np.concatenate([np.median(shares, axis=0), np.percentile(shares, 95, axis=0)])
