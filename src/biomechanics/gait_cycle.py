import numpy as np
from scipy.signal import find_peaks


def detect_initial_contacts(
    df,
    side,
    fps,
    min_interval_sec=0.55,
    prominence=0.015,
):
    """
    Detect candidate Initial Contact events from the foot trajectory.

    Uses both heel and foot-index vertical coordinates.
    In image coordinates, larger Y means lower position.
    """

    toe_col = f"{side}_foot_index_smooth_y"
    heel_col = f"{side}_heel_smooth_y"

    if toe_col not in df.columns:
        raise KeyError(f"Column not found: {toe_col}")

    if heel_col not in df.columns:
        raise KeyError(f"Column not found: {heel_col}")

    toe_y = df[toe_col].to_numpy(dtype=float)
    heel_y = df[heel_col].to_numpy(dtype=float)

    valid_toe = np.isfinite(toe_y)
    valid_heel = np.isfinite(heel_y)

    if valid_toe.sum() < 5 or valid_heel.sum() < 5:
        return np.array([], dtype=int)

    frames = np.arange(len(df))

    toe_y = np.interp(
        frames,
        frames[valid_toe],
        toe_y[valid_toe],
    )

    heel_y = np.interp(
        frames,
        frames[valid_heel],
        heel_y[valid_heel],
    )

    # Lower point of the foot in image coordinates.
    foot_y = np.maximum(toe_y, heel_y)

    min_distance = max(
        1,
        int(round(fps * min_interval_sec))
    )

    peaks, _ = find_peaks(
        foot_y,
        distance=min_distance,
        prominence=prominence,
    )

    return peaks

import numpy as np


def detect_toe_off(
    df,
    side,
    ic_frames,
    fps,
    height,
    min_phase_fraction=0.20,
    max_phase_fraction=0.65,
    lookahead_sec=0.10,
    lift_threshold_px=6.0,
):
    """
    Detect candidate Toe-Off events between consecutive ICs.

    Uses the smoothed foot-index (toe) vertical trajectory.
    Image Y decreases when the toe moves upward.

    This is a kinematic heuristic, not direct proof of ground contact.
    """

    if fps <= 0 or height <= 0:
        return np.array([], dtype=int)

    col = f"{side}_foot_index_smooth_y"

    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")

    y = df[col].to_numpy(dtype=float)

    valid = np.isfinite(y)

    if valid.sum() < 5:
        return np.array([], dtype=int)

    # Fill missing values and convert normalized Y to pixels.
    y = np.interp(
        np.arange(len(y)),
        np.flatnonzero(valid),
        y[valid],
    )
    y *= height

    ic_frames = np.asarray(ic_frames, dtype=int)
    ic_frames = np.unique(
        ic_frames[
            (ic_frames >= 0) &
            (ic_frames < len(df))
        ]
    )

    if len(ic_frames) < 2:
        return np.array([], dtype=int)

    lookahead = max(2, round(fps * lookahead_sec))
    toe_offs = []

    for ic_start, next_ic in zip(
        ic_frames[:-1],
        ic_frames[1:],
    ):
        cycle_length = next_ic - ic_start

        # Limit the search to a plausible part of the cycle.
        search_start = ic_start + max(
            2,
            round(cycle_length * min_phase_fraction),
        )

        search_end = min(
            next_ic - lookahead - 1,
            ic_start + round(
                cycle_length * max_phase_fraction
            ),
        )

        if search_end < search_start:
            continue

        scores = []

        # Positive score means the toe moves upward
        # over the following lookahead window.
        for frame in range(search_start, search_end + 1):
            future = y[frame + 1:frame + lookahead + 1]

            if len(future) < lookahead:
                scores.append(np.nan)
                continue

            score = y[frame] - np.median(future)
            scores.append(score)

        scores = np.asarray(scores, dtype=float)
        finite = np.isfinite(scores)

        if not finite.any():
            continue

        best_score = np.max(scores[finite])

        # Not enough vertical movement to identify a candidate.
        if best_score < lift_threshold_px:
            continue

        # Select the beginning of the strongest sustained lift,
        # rather than automatically choosing the first allowed frame.
        target_score = max(
            lift_threshold_px,
            0.70 * best_score,
        )

        candidates = np.flatnonzero(
            finite & (scores >= target_score)
        )

        if len(candidates) == 0:
            continue

        toe_offs.append(
            search_start + int(candidates[0])
        )

    return np.asarray(toe_offs, dtype=int)