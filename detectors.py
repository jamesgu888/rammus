import numpy as np

neighbours = 8  # how many healthy recordings at the closest shaft speeds to compare a clip with


def split_healthy(healthy_recordings, seed):
    # 60% reference, 20% calibration, 20% test, split by whole recording
    shuffled = np.random.default_rng(seed).permutation(healthy_recordings)
    n_reference = int(0.6 * len(shuffled))
    n_calibration = int(0.2 * len(shuffled))
    reference = shuffled[:n_reference]
    calibration = shuffled[n_reference:n_reference + n_calibration]
    test = shuffled[n_reference + n_calibration:]
    return reference, calibration, test


def z_score(reference, value):
    # how many standard deviations each measurement is from healthy; the biggest one is the score
    mean = np.mean(reference, axis=0)
    std = np.std(reference, axis=0)
    keep = std > 0
    z = np.abs(value[keep] - mean[keep]) / std[keep]
    return np.max(z), np.flatnonzero(keep)[np.argmax(z)]


def score_clips(values, clips, clip_recording, recording_speed, reference, speed_matched):
    # Score each clip against the healthy reference recordings.
    # Speed-matched only uses the reference recordings closest to the clip's shaft speed,
    # because a healthy machine sounds different at different speeds.
    scores = []
    flagged = []
    for i in clips:
        use = reference
        if speed_matched:
            distance = np.abs(recording_speed[reference] - recording_speed[clip_recording[i]])
            use = reference[np.argsort(distance)[:neighbours]]
        score, feature = z_score(values[np.isin(clip_recording, use)], values[i])
        scores.append(score)
        flagged.append(feature)
    return np.array(scores), np.array(flagged)


def auroc(healthy_scores, faulty_scores):
    # fraction of (faulty, healthy) pairs where the faulty clip scores higher; ties count half
    higher = faulty_scores[:, None] > healthy_scores[None, :]
    tied = faulty_scores[:, None] == healthy_scores[None, :]
    return np.mean(higher) + 0.5 * np.mean(tied)
