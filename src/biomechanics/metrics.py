import numpy as np
from scipy.signal import savgol_filter


def angle(a, b, c):
    # angle ABC in degrees
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def _smooth(x, window=9, poly=2):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 5:
        return x
    xi = np.arange(len(x))
    filled = np.interp(xi, xi[valid], x[valid])
    w = min(window, len(xi) if len(xi) % 2 else len(xi)-1)
    if w < 5:
        return filled
    return savgol_filter(filled, w, poly)


def compute_metrics(df, width, height):
    out = df.copy()

    for side in ["left", "right"]:
        hip = out[
            [f"{side}_hip_smooth_x", f"{side}_hip_smooth_y"]
        ].values.copy()

        knee = out[
            [f"{side}_knee_smooth_x", f"{side}_knee_smooth_y"]
        ].values.copy()

        ankle = out[
            [f"{side}_ankle_smooth_x", f"{side}_ankle_smooth_y"]
        ].values.copy()

        shoulder = out[
            [f"{side}_shoulder_x", f"{side}_shoulder_y"]
        ].values.copy()

        foot = out[
            [f"{side}_foot_index_smooth_x",
            f"{side}_foot_index_smooth_y"]
        ].values.copy()

        # MediaPipe coordinates are normalized [0, 1].
        # Convert them to pixel coordinates.
        for points in [hip, knee, ankle, shoulder, foot]:
            points[:, 0] *= width
            points[:, 1] *= height

        # Geometric knee angle: HIP-KNEE-ANKLE
        out[f"{side}_knee_angle"] = [
            angle(h, k, a)
            for h, k, a in zip(hip, knee, ankle)
        ]

        # Geometric hip angle: SHOULDER-HIP-KNEE
        out[f"{side}_hip_angle"] = [
            angle(s, h, k)
            for s, h, k in zip(shoulder, hip, knee)
        ]

        # Geometric ankle angle: KNEE-ANKLE-FOOT
        out[f"{side}_ankle_angle"] = [
            angle(k, a, f)
            for k, a, f in zip(knee, ankle, foot)
        ]

        out[f"{side}_knee_angle_smooth"] = _smooth(
            out[f"{side}_knee_angle"]
        )

    # Mid-hip and shoulder line
    mid_hip_x = (out.left_hip_x + out.right_hip_x) / 2
    mid_hip_y = (out.left_hip_y + out.right_hip_y) / 2
    mid_sh_x = (out.left_shoulder_x + out.right_shoulder_x) / 2
    mid_sh_y = (out.left_shoulder_y + out.right_shoulder_y) / 2

    dx = mid_sh_x - mid_hip_x
    dy = mid_sh_y - mid_hip_y

    out["trunk_inclination_deg"] = np.degrees(
        np.arctan2(dx, -dy)
    )

    out["pelvis_y"] = mid_hip_y

    return out


def smooth_landmarks(df, fps, window_sec=0.15, poly=2):
    """
    Smooth 2D landmark trajectories with Savitzky-Golay filter.

    Raw coordinates are preserved in the original columns.
    Filtered coordinates are stored in *_smooth_x / *_smooth_y.
    """

    out = df.copy()

    landmarks = [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    if fps <= 0 or len(out) < 5:
        return out

    # About 150 ms of signal.
    window = int(round(fps * window_sec))

    # Savitzky-Golay requires an odd window.
    if window % 2 == 0:
        window += 1

    window = max(5, window)

    # Window cannot be larger than the signal.
    max_window = len(out) if len(out) % 2 == 1 else len(out) - 1
    window = min(window, max_window)

    if window <= poly:
        return out

    for name in landmarks:
        for axis in ["x", "y"]:
            col = f"{name}_{axis}"
            smooth_col = f"{name}_smooth_{axis}"

            if col not in out:
                continue

            x = out[col].to_numpy(dtype=float)

            valid = np.isfinite(x)

            if valid.sum() < window:
                out[smooth_col] = x
                continue

            # Interpolate missing values.
            indices = np.arange(len(x))
            filled = np.interp(
                indices,
                indices[valid],
                x[valid],
            )

            # Savitzky-Golay smoothing.
            filtered = savgol_filter(
                filled,
                window_length=window,
                polyorder=poly,
            )

            out[smooth_col] = filtered

    return out
