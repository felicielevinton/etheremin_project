"""Detect immobility periods and describe movement bouts for one session."""

import numpy as np
import pandas as pd


def detect_movement_bouts(
    session_features,
    info,
    *,
    dt,
    speed_x_factor,
    velocity_threshold,
    min_immobility_s,
):
    """Return movement_bouts and immobility_bouts DataFrames for one session.

    info supplies animal, headstage, and session identifiers. Speed_x is scaled
    by speed_x_factor before thresholding. Bins are positional row offsets.
    Immobility stop bins are exclusive and identify movement onset when a
    following sample exists. Movement ends at the next qualifying immobility
    start. Unknown endpoints and their dependent measurements remain missing.
    Condition is read at onset, including bouts crossing condition boundaries.
    """
    position = session_features["Position"].to_numpy()
    speed = session_features["Speed_x"].to_numpy() * speed_x_factor

    # Find continuous immobility segments.

    is_immobile_instantaneous = np.abs(speed) < velocity_threshold

    # Locate transitions, including runs at either end of the session.
    changes = np.diff(np.pad(is_immobile_instantaneous.astype(int), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    # Retain short segments as well for plotting.
    immobility_bouts = pd.DataFrame({
        "start_bin": starts,
        "stop_bin": stops,
        "duration_s": (stops - starts) * dt,
    })

    immobility_bouts["qualifies"] = (
        immobility_bouts["duration_s"] >= min_immobility_s
    )

    # Pair each onset with the next qualifying immobility start.
    quiet_bouts = immobility_bouts.loc[immobility_bouts["qualifies"]]

    movement_starts = quiet_bouts["stop_bin"].to_numpy()
    movement_ends = np.full(len(movement_starts), np.nan)
    movement_ends[:-1] = quiet_bouts["start_bin"].to_numpy()[1:]

    # An immobility period reaching the session end has no following movement so exclude it
    has_onset = movement_starts < len(position)
    movement_starts = movement_starts[has_onset]
    movement_ends = movement_ends[has_onset]

    # One row per movement onset, including bouts without a known endpoint.
    movement_bouts = pd.DataFrame({
        "animal": info["animal"],
        "headstage": info["headstage"],
        "session": info["session"],
        "condition": session_features["Condition"].iloc[movement_starts].to_numpy(),
        "onset_bin": movement_starts,
        "onset_time_s": movement_starts * dt,
        "end_bin": pd.array(movement_ends, dtype="Int64"),
        "end_time_s": movement_ends * dt,
        "initial_position": position[movement_starts],
        "initial_direction": np.sign(
            position[movement_starts] - position[movement_starts - 1]
        ),
    })

    # Endpoint-dependent measurements stay missing for unfinished bouts.
    movement_bouts["final_position"] = np.nan
    movement_bouts["total_movement"] = np.nan
    movement_bouts["mean_speed"] = np.nan

    for bout_idx, (onset, end) in enumerate(zip(movement_starts, movement_ends)):
        if np.isfinite(end):
            end = int(end)
            movement_bouts.loc[bout_idx, "final_position"] = position[end]
            # Total distance travelled includes reversals and the endpoint sample.
            movement_bouts.loc[bout_idx, "total_movement"] = np.abs(
                np.diff(position[onset:end + 1])
            ).sum()
            movement_bouts.loc[bout_idx, "mean_speed"] = speed[onset:end].mean()

    return movement_bouts, immobility_bouts
