"""
Loads the MAFAULDA data.

Downloads the recordings the first time, keeps only the microphone channel, and cuts each
5 s recording into 1 s clips with both detectors' features measured on each clip.
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.signal import resample_poly

from rammus import rammus_features, sample_rate
from rms import rms_features

# MAFAULDA (www02.smt.ufrj.br/~offshore/mfs/page_01.html): a motor turning a shaft on two bearings,
# recorded for 5 s at 50 kHz. Each recording is a CSV whose 8th column is a microphone.
url = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/"
conditions = ["normal", "imbalance", "horizontal-misalignment", "overhang"]
data_dir = Path(__file__).parent / "data"
cache = data_dir / "mafaulda_microphone_25k.npz"


def load_recordings():
    # The first run downloads 7 GB of zips, keeps only the microphone signal, and deletes the zips.
    if cache.exists():
        saved = np.load(cache)
        return saved["names"], saved["signals"]

    data_dir.mkdir(exist_ok=True)
    names = []
    signals = []
    for condition in conditions:
        path = data_dir / f"{condition}.zip"
        if not path.exists():
            print(f"Downloading {condition}.zip...")
            urlretrieve(url + f"{condition}.zip", path)
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
                with z.open(name) as csv:
                    s = np.loadtxt(csv, delimiter=",", usecols=7)
                names.append(name)
                # 50 kHz -> 25 kHz, which also filters out a loud 22 kHz tone from the motor drive
                signals.append(resample_poly(s, 1, 2).astype(np.float32))
        path.unlink()

    np.savez(cache, names=np.array(names), signals=np.array(signals))
    return np.array(names), np.array(signals)


def fault_type(name):
    # "normal/12.288.csv" -> "normal", "overhang/cage_fault/6g/12.288.csv" -> "bearing cage_fault"
    parts = name.split("/")
    if parts[0] == "overhang":
        return "bearing " + parts[1]
    return parts[0]


def shaft_speed(name):
    # the file name is the shaft speed in Hz
    return float(Path(name).stem)


def cut_clips(signals):
    # cut each 5 s recording into five 1 s clips and measure both detectors' features on each
    clip_rms = []
    clip_rammus = []
    clip_recording = []
    for i, s in enumerate(signals):
        for start in range(0, len(s) - sample_rate + 1, sample_rate):
            clip = s[start:start + sample_rate].astype(float)
            clip_rms.append(rms_features(clip))
            clip_rammus.append(rammus_features(clip))
            clip_recording.append(i)
    return np.array(clip_rms), np.array(clip_rammus), np.array(clip_recording)
