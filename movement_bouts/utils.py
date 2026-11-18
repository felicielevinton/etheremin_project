"""Movement-bout detection and extraction of aligned premotor spike counts."""

import pickle
from pathlib import Path

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


def extract_premotor_bouts(neural_data, session_bouts, *, dt, premotor_duration_s):
    """Extract one session into a table keyed by the original bout index.

    Each spike_counts cell is an independent (neurons, time bins) array for
    [-premotor_duration_s, 0), retaining the original counts and bin width.
    Return (neuronal_bouts, excluded_bouts). The latter contains only reasons;
    all behavioral metadata remains in the original movement_bouts table.
    """
    if dt <= 0 or premotor_duration_s <= 0:
        raise ValueError("Bin width and premotor duration must be positive.")
    n_bins = round(premotor_duration_s / dt)
    if n_bins < 1 or not np.isclose(n_bins * dt, premotor_duration_s, rtol=0, atol=1e-10):
        raise ValueError("Premotor duration must contain a whole number of bins.")
    if neural_data.ndim != 2:
        raise ValueError("Neural data must have shape (neurons, time bins).")
    if not session_bouts.index.is_unique:
        raise ValueError("Movement-bout indices must be unique.")

    onset_bins = session_bouts["onset_bin"].to_numpy(dtype=int)
    start_bins = onset_bins - n_bins
    valid = (start_bins >= 0) & (onset_bins < neural_data.shape[1])

    # Copy slices so the extracted windows do not retain the full recording.
    neuronal_bouts = pd.DataFrame(
        {"spike_counts": [
            neural_data[:, start:onset].copy()
            for start, onset in zip(start_bins[valid], onset_bins[valid])
        ]},
        index=session_bouts.index[valid],
    )
    excluded_bouts = pd.DataFrame(
        {"exclusion_reason": np.where(
            start_bins[~valid] < 0,
            "Premotor window starts before recording",
            "Movement onset is outside recording",
        )},
        index=session_bouts.index[~valid],
    )
    return neuronal_bouts, excluded_bouts


def extract_all_premotor_bouts(
    data_dir, movement_bouts, session_info, *, dt, premotor_duration_s
):
    """Load collections once and extract their sessions one at a time.

    Return neuronal_bouts, neuron_ids_by_session, and excluded_bouts.
    Bout tables retain the movement_bouts index. Neuron IDs use the
    session_info index (idx). Pass session_info.loc[[idx]] for one session.
    """
    if not movement_bouts.index.is_unique or not session_info.index.is_unique:
        raise ValueError("Bout and session indices must be unique.")
    data_dir = Path(data_dir)
    bout_tables = []
    excluded_tables = []
    session_neuron_ids = {}

    for feature_file, sessions in session_info.groupby("feature_file", sort=False):
        neural_file = feature_file.replace("_feature_ss", "_data_ss")
        neuron_file = feature_file.replace("_feature_ss", "_neuron_idx_ss")
        with (data_dir / neural_file).open("rb") as f:
            neural_collection = pickle.load(f)
        with (data_dir / neuron_file).open("rb") as f:
            neuron_collection = pickle.load(f)

        for idx, info in sessions.iterrows():
            session = int(info["session"])
            neural_data = neural_collection[session]
            neuron_ids = neuron_collection[session]
            if neural_data.shape != (len(neuron_ids), int(info["n_bins"])):
                raise ValueError(f"Neural dimensions do not match session_info for idx={idx}.")
            session_bouts = movement_bouts.loc[movement_bouts["idx"] == idx]
            bouts, excluded = extract_premotor_bouts(
                neural_data, session_bouts, dt=dt, premotor_duration_s=premotor_duration_s
            )
            bout_tables.append(bouts)
            excluded_tables.append(excluded)
            session_neuron_ids[idx] = neuron_ids
        del neural_data, neural_collection, neuron_collection
        print(f"{neural_file}: extracted {len(sessions)} sessions")

    neuronal_bouts = (
        pd.concat(bout_tables) if bout_tables else pd.DataFrame(columns=["spike_counts"])
    )
    excluded_bouts = (
        pd.concat(excluded_tables) if excluded_tables else pd.DataFrame(columns=["exclusion_reason"])
    )
    # Preserve source-table order as well as its identifiers.
    neuronal_bouts = neuronal_bouts.reindex(
        movement_bouts.index[movement_bouts.index.isin(neuronal_bouts.index)]
    )
    excluded_bouts = excluded_bouts.reindex(
        movement_bouts.index[movement_bouts.index.isin(excluded_bouts.index)]
    )
    neuron_ids_by_session = pd.DataFrame(
        {"neuron_ids": pd.Series(session_neuron_ids, dtype=object)}
    ).reindex(session_info.index)
    return neuronal_bouts, neuron_ids_by_session, excluded_bouts


def prepare_pca_input(
    neuronal_bouts, movement_bouts, neuron_ids_by_session, *, idx, dt, n_bins
):
    """Average one session's counts and return aligned neural/metadata tables.

    n_bins is the number of equal windows spanning the saved premotor period.
    Return pca_input (observations x neurons, spikes/s) and pca_labels (all
    original bout metadata, bin_number, bin_end_s). Both share the unique index
    (bout_id, bin_start_s), ordered by bout then increasing window time.
    Bout-level labels are repeated, not recomputed within each window.
    This function performs no file I/O, standardization, or PCA fitting.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive bin width in seconds.")

    session_bouts = movement_bouts.loc[movement_bouts["idx"] == idx].join(
        neuronal_bouts, how="inner", validate="one_to_one"
    )
    if session_bouts.empty:
        raise ValueError(f"No extracted bouts are available for idx={idx}.")
    neuron_ids = neuron_ids_by_session.loc[idx, "neuron_ids"]
    spike_counts = np.stack(session_bouts["spike_counts"].to_numpy())
    n_bouts, n_neurons, n_time_bins = spike_counts.shape
    if n_time_bins % n_bins != 0:
        raise ValueError("n_bins must divide the number of original time bins.")

    bins_per_window = n_time_bins // n_bins
    window_s = bins_per_window * dt
    firing_rates = spike_counts.reshape(
        n_bouts, n_neurons, n_bins, bins_per_window
    ).sum(axis=-1) / window_s
    bin_edges_s = np.round(np.arange(n_bins + 1) * window_s - n_time_bins * dt, 12)

    repeated_bout_ids = np.repeat(session_bouts.index.to_numpy(), n_bins)
    observation_index = pd.MultiIndex.from_arrays(
        [repeated_bout_ids, np.tile(bin_edges_s[:-1], n_bouts)],
        names=["bout_id", "bin_start_s"],
    )
    pca_input = pd.DataFrame(
        firing_rates.transpose(0, 2, 1).reshape(n_bouts * n_bins, n_neurons),
        index=observation_index,
        columns=pd.Index(neuron_ids, name="neuron_id"),
    )
    pca_labels = session_bouts.drop(columns="spike_counts").loc[repeated_bout_ids].copy()
    pca_labels.index = observation_index
    pca_labels["bin_number"] = np.tile(np.arange(n_bins), n_bouts)
    pca_labels["bin_end_s"] = np.tile(bin_edges_s[1:], n_bouts)
    return pca_input, pca_labels
