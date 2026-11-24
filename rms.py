"""
The baseline detector's feature: how loud the clip is (its root-mean-square amplitude).
"""

import numpy as np


def rms_features(s):
    # the baseline detector: one number per clip, the root-mean-square amplitude (how loud the clip is)
    return np.array([np.sqrt(np.mean(s ** 2))])
