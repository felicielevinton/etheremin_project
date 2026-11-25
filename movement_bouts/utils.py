"""Movement-bout detection and extraction of aligned premotor spike counts."""

import hashlib
import json
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
    early_speed_window_s=1.0,
):
    """Return movement_bouts and immobility_bouts DataFrames for one session.

    info supplies animal, headstage, and session identifiers. Speed_x is scaled
    by speed_x_factor before thresholding. Bins are positional row offsets.
    Immobility stop bins are exclusive and identify movement onset when a
    following sample exists. Movement ends at the next qualifying immobility
    start. Unknown endpoints and their dependent measurements remain missing.
    Condition is read at onset, including bouts crossing condition boundaries.
    Early mean/peak speed use [onset, min(onset + early_speed_window_s, end)).
    End is the next qualifying immobility start, as for the full bout. An
    unfinished bout can have early measurements if the complete requested
    interval is recorded; recording-truncated intervals have missing speeds.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite bin width in seconds.")
    if not np.isfinite(early_speed_window_s) or early_speed_window_s <= 0:
        raise ValueError("early_speed_window_s must be positive and finite.")
    early_bins = round(early_speed_window_s / dt)
    if early_bins < 1 or not np.isclose(early_bins * dt, early_speed_window_s, rtol=0, atol=1e-10):
        raise ValueError("early_speed_window_s must align with the original time bins.")
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
    movement_bouts["early_mean_speed"] = np.nan
    movement_bouts["peak_speed"] = np.nan
    movement_bouts["peak_speed_latency_s"] = np.nan
    movement_bouts["early_speed_duration_s"] = np.nan
    movement_bouts["early_speed_complete"] = False

    for bout_idx, (onset, end) in enumerate(zip(movement_starts, movement_ends)):
        requested_stop = onset + early_bins
        if np.isfinite(end):
            requested_stop = min(requested_stop, int(end))
        observed_stop = min(requested_stop, len(speed))
        movement_bouts.loc[bout_idx, "early_speed_duration_s"] = (observed_stop - onset) * dt
        complete = observed_stop == requested_stop and observed_stop > onset
        movement_bouts.loc[bout_idx, "early_speed_complete"] = complete
        early_speed = np.abs(speed[onset:observed_stop])
        if complete and np.isfinite(early_speed).all():
            peak_offset = int(np.argmax(early_speed))
            movement_bouts.loc[bout_idx, "early_mean_speed"] = early_speed.mean()
            movement_bouts.loc[bout_idx, "peak_speed"] = early_speed[peak_offset]
            movement_bouts.loc[bout_idx, "peak_speed_latency_s"] = peak_offset * dt
        if np.isfinite(end):
            end = int(end)
            movement_bouts.loc[bout_idx, "final_position"] = position[end]
            # Total distance travelled includes reversals and the endpoint sample.
            movement_bouts.loc[bout_idx, "total_movement"] = np.abs(
                np.diff(position[onset:end + 1])
            ).sum()
            movement_bouts.loc[bout_idx, "mean_speed"] = speed[onset:end].mean()

    return movement_bouts, immobility_bouts


def save_movement_bout_tables(movement_bouts, immobility_bouts, session_info, parameters, output_dir):
    """Save the reference tables, retaining earlier bout bytes for provenance.

    Unchanged tables keep their original file bytes and hashes. A replaced bout
    table is archived by SHA-256 so existing analyses can verify that an update
    only adds columns, rather than changing their original observations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("movement_bouts.pkl", movement_bouts),
        ("immobility_bouts.pkl", immobility_bouts),
        ("session_info.pkl", session_info),
    ):
        path = output_dir / name
        if path.exists():
            previous = pd.read_pickle(path)
            if previous.equals(table):
                continue
            if name == "movement_bouts.pkl":
                previous_bytes = path.read_bytes()
                digest = hashlib.sha256(previous_bytes).hexdigest()
                archive = output_dir / "source_versions" / f"{digest}.pkl"
                archive.parent.mkdir(exist_ok=True)
                if archive.exists():
                    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Archived bout source is corrupt: {archive}")
                else:
                    archive.write_bytes(previous_bytes)
                previous_parameters = output_dir / "parameters.json"
                archived_parameters = archive.with_suffix(".parameters.json")
                if previous_parameters.exists() and not archived_parameters.exists():
                    archived_parameters.write_bytes(previous_parameters.read_bytes())
        temporary = path.with_suffix(".pkl.tmp")
        table.to_pickle(temporary)
        temporary.replace(path)
    parameter_path = output_dir / "parameters.json"
    parameter_temporary = parameter_path.with_suffix(".json.tmp")
    parameter_temporary.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    parameter_temporary.replace(parameter_path)


def validate_bout_sources(bout_dir, source_sha256):
    """Verify exact sources, allowing an audited column-only bout extension.

    Old hashes remain meaningful: an archived pickle must match the expected
    hash, and every original column, value, dtype and row/index order must still
    match the current table. Session metadata always requires its exact hash.
    """
    bout_dir = Path(bout_dir)
    for filename, expected in source_sha256.items():
        path = bout_dir / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        if filename == "movement_bouts.pkl" and len(expected) == 64 and all(
            character in "0123456789abcdef" for character in expected
        ):
            archive = bout_dir / "source_versions" / f"{expected}.pkl"
            if archive.is_file() and hashlib.sha256(archive.read_bytes()).hexdigest() == expected:
                previous, current = pd.read_pickle(archive), pd.read_pickle(path)
                if (previous.index.identical(current.index)
                        and previous.columns.isin(current.columns).all()
                        and previous.equals(current.loc[:, previous.columns])):
                    continue
        raise ValueError(f"{filename} differs from the saved source; regenerate aligned results.")


def _window_bins(window_s, dt, name):
    """Convert a finite, ordered premotor interval to exact original-bin offsets."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite bin width in seconds.")
    edges = np.asarray(window_s, dtype=float)
    if edges.shape != (2,) or not np.isfinite(edges).all():
        raise ValueError(f"{name} must contain two finite times in seconds.")
    if edges[0] >= edges[1] or edges[1] > 0:
        raise ValueError(f"{name} must have start < end <= 0.")
    offsets = np.rint(edges / dt).astype(int)
    if not np.allclose(offsets * dt, edges, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must align with the original time bins.")
    return offsets


def extract_premotor_bouts(neural_data, session_bouts, *, dt, extraction_window_s):
    """Extract one session into a table keyed by the original bout index.

    Each spike_counts cell is an independent (neurons, time bins) array for
    extraction_window_s, start-inclusive and end-exclusive relative to onset.
    Retain original counts and bin width; analysis/baseline selection comes later.
    Return (neuronal_bouts, excluded_bouts). The latter contains only reasons;
    all behavioral metadata remains in the original movement_bouts table.
    """
    offsets = _window_bins(extraction_window_s, dt, "extraction_window_s")
    if neural_data.ndim != 2:
        raise ValueError("Neural data must have shape (neurons, time bins).")
    if not session_bouts.index.is_unique:
        raise ValueError("Movement-bout indices must be unique.")

    onset_bins = session_bouts["onset_bin"].to_numpy(dtype=int)
    start_bins = onset_bins + offsets[0]
    stop_bins = onset_bins + offsets[1]
    valid = (start_bins >= 0) & (onset_bins >= 0) & (onset_bins < neural_data.shape[1])

    # Copy slices so the extracted windows do not retain the full recording.
    neuronal_bouts = pd.DataFrame(
        {"spike_counts": [
            neural_data[:, start:stop].copy()
            for start, stop in zip(start_bins[valid], stop_bins[valid])
        ]},
        index=session_bouts.index[valid],
    )
    excluded_bouts = pd.DataFrame(
        {"exclusion_reason": np.where(
            start_bins[~valid] < 0,
            "Extraction window starts before recording",
            "Movement onset is outside recording",
        )},
        index=session_bouts.index[~valid],
    )
    return neuronal_bouts, excluded_bouts


def extract_all_premotor_bouts(
    data_dir, movement_bouts, session_info, *, dt, extraction_window_s
):
    """Load collections once and extract their sessions one at a time.

    Return neuronal_bouts, neuron_ids_by_session, and excluded_bouts.
    Bout tables retain the movement_bouts index. Neuron IDs use the
    session_info index (idx). Pass session_info.loc[[idx]] for one session.
    """
    _window_bins(extraction_window_s, dt, "extraction_window_s")
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
                neural_data, session_bouts, dt=dt, extraction_window_s=extraction_window_s
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
    neuronal_bouts, movement_bouts, neuron_ids_by_session, *, idx, dt, n_bins,
    extraction_window_s, analysis_window_s, baseline_window_s=None, activity_mode="raw",
):
    """Average one session's counts and return aligned neural/metadata tables.

    Slice analysis_window_s from counts covering the saved extraction_window_s.
    n_bins is the number of equal averaging windows within the analysis interval.
    In baseline_subtracted mode, subtract each bout/neuron's mean baseline firing
    rate from all its analysis windows. Both raw/corrected values are in spikes/s.
    Return pca_input (observations x neurons) and pca_labels (all
    original bout metadata, bin_number, bin_end_s). Both share the unique index
    (bout_id, bin_start_s), ordered by bout then increasing window time.
    Bout-level labels are repeated, not recomputed within each window.
    This function performs no file I/O, variance standardization, or PCA fitting.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    extraction_edges = _window_bins(extraction_window_s, dt, "extraction_window_s")
    analysis_edges = _window_bins(analysis_window_s, dt, "analysis_window_s")
    if analysis_edges[0] < extraction_edges[0] or analysis_edges[1] > extraction_edges[1]:
        raise ValueError("Analysis window must be contained in the saved extraction.")
    if activity_mode not in ("raw", "baseline_subtracted"):
        raise ValueError("activity_mode must be 'raw' or 'baseline_subtracted'.")
    baseline_edges = None
    if activity_mode == "baseline_subtracted":
        baseline_edges = _window_bins(baseline_window_s, dt, "baseline_window_s")
        if baseline_edges[0] < extraction_edges[0] or baseline_edges[1] > extraction_edges[1]:
            raise ValueError("Baseline window must be contained in the saved extraction.")
        if baseline_edges[1] > analysis_edges[0]:
            raise ValueError("Baseline window must end at or before the analysis window starts.")

    session_bouts = movement_bouts.loc[movement_bouts["idx"] == idx].join(
        neuronal_bouts, how="inner", validate="one_to_one"
    )
    if session_bouts.empty:
        raise ValueError(f"No extracted bouts are available for idx={idx}.")
    neuron_ids = neuron_ids_by_session.loc[idx, "neuron_ids"]
    spike_counts = np.stack(session_bouts["spike_counts"].to_numpy())
    n_bouts, n_neurons, n_time_bins = spike_counts.shape
    if n_time_bins != extraction_edges[1] - extraction_edges[0]:
        raise ValueError("Saved array length does not match extraction_window_s and dt.")
    if n_neurons != len(neuron_ids):
        raise ValueError("Saved neuron IDs do not match the neural array's neuron axis.")
    analysis_counts = spike_counts[:, :,
        analysis_edges[0] - extraction_edges[0]:analysis_edges[1] - extraction_edges[0]
    ]
    n_analysis_bins = analysis_counts.shape[-1]
    if n_analysis_bins % n_bins != 0:
        raise ValueError("n_bins must divide the number of original analysis bins.")

    bins_per_window = n_analysis_bins // n_bins
    window_s = bins_per_window * dt
    firing_rates = analysis_counts.reshape(
        n_bouts, n_neurons, n_bins, bins_per_window
    ).sum(axis=-1) / window_s
    if baseline_edges is not None:
        baseline_counts = spike_counts[:, :,
            baseline_edges[0] - extraction_edges[0]:baseline_edges[1] - extraction_edges[0]
        ]
        baseline_rates = baseline_counts.sum(axis=-1) / ((baseline_edges[1] - baseline_edges[0]) * dt)
        firing_rates -= baseline_rates[:, :, None]
    bin_edges_s = np.round(np.arange(n_bins + 1) * window_s + analysis_edges[0] * dt, 12)

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


def fit_pca_by_condition(pca_input, pca_labels):
    """Fit independent tracking/playback PCAs, retaining models and labeled scores.

    Return a dictionary keyed by condition (0/1). Conditions with fewer than
    two observations are omitted. Each model retains all components, centers
    its own condition's rates, and does not standardize or whiten them.
    """
    from sklearn.decomposition import PCA

    fits = {}
    for condition in (0, 1):
        condition_input = pca_input.loc[pca_labels["condition"] == condition]
        if len(condition_input) < 2:
            continue
        model = PCA(svd_solver="full")
        scores = pd.DataFrame(
            model.fit_transform(condition_input),
            index=condition_input.index,
            columns=[f"PC{i + 1}" for i in range(model.n_components_)],
        )
        fits[condition] = {
            "model": model,
            "results": pca_labels.loc[condition_input.index].join(scores, validate="one_to_one"),
        }
    return fits


def project_onto_tracking_pca(pca_input, pca_labels, tracking_model):
    """Transform both conditions with a saved tracking PCA, without refitting.

    transform() uses the tracking mean for both conditions, preserving any
    playback mean offset. Neuron columns follow the tracking model's order.
    Return the same condition/model/results structure as fit_pca_by_condition().
    """
    projections = {}
    for condition in (0, 1):
        observations = pca_input.loc[
            pca_labels["condition"] == condition, tracking_model.feature_names_in_
        ]
        if observations.empty:
            continue
        scores = pd.DataFrame(
            tracking_model.transform(observations),
            index=observations.index,
            columns=[f"PC{i + 1}" for i in range(tracking_model.n_components_)],
        )
        projections[condition] = {
            "model": tracking_model,
            "results": pca_labels.loc[observations.index].join(scores, validate="one_to_one"),
        }
    return projections


def plot_pca_session(fits, *, title, position_limits=None):
    """Draw saved PC1/PC2/PC3 scores without preparing data or fitting models.

    Use shared position_limits across sessions for a consistent color scale.
    Return the Matplotlib figure and axes; the caller displays/closes them.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    # Use one starting-position color scale across the two independent PCA panels.
    positions = pd.concat([fit["results"]["initial_position"] for fit in fits.values()])
    if position_limits is None:
        position_limits = (positions.min(), positions.max())
    position_norm = Normalize(*position_limits)
    position_cmap = "viridis"
    directions = ((-1, "x", "Left"), (1, "o", "Right"), (0, "s", "Zero / unknown direction"))

    fig_pca, axes_pca = plt.subplots(
        1, 2, figsize=(14, 6), layout="constrained",
        subplot_kw={"projection": "3d"},
    )
    for ax, (condition, name) in zip(axes_pca, ((0, "Tracking"), (1, "Playback"))):
        if condition not in fits or fits[condition]["model"].n_components_ < 3:
            ax.text2D(0.5, 0.5, "Fewer than three components available", ha="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        model = fits[condition]["model"]
        rows = fits[condition]["results"]
        legend_handles = []
        for direction, marker, label in directions:
            mask = rows["initial_direction"] == direction
            if direction == 0:
                mask = mask | rows["initial_direction"].isna()
            subset = rows.loc[mask]
            if subset.empty:
                continue
            ax.scatter(
                subset["PC1"], subset["PC2"], subset["PC3"],
                c=subset["initial_position"], cmap=position_cmap, norm=position_norm,
                marker=marker, s=25, alpha=0.7, linewidths=0.8, depthshade=False,
            )
            legend_handles.append(Line2D(
                [], [], color="black", marker=marker, linestyle="none",
                markersize=6, label=label,
            ))

        ax.set(
            xlabel=f"PC1 ({model.explained_variance_ratio_[0]:.1%} variance)",
            ylabel=f"PC2 ({model.explained_variance_ratio_[1]:.1%} variance)",
            zlabel=f"PC3 ({model.explained_variance_ratio_[2]:.1%} variance)",
            title=f"{name}: separate PCA ({len(rows)} observations)",
        )
        ax.set_box_aspect(np.maximum(np.ptp(rows[["PC1", "PC2", "PC3"]].to_numpy(), axis=0), 1e-9))
        ax.view_init(elev=22, azim=-60)
        ax.legend(handles=legend_handles, frameon=False)

    fig_pca.colorbar(
        plt.cm.ScalarMappable(norm=position_norm, cmap=position_cmap),
        ax=axes_pca, label="Starting position", shrink=0.85,
    )
    fig_pca.suptitle(title + " | Each point = one bout at one time window")
    return fig_pca, axes_pca


def _link_pca_cameras(figure, scene_names=("scene", "scene2")):
    """Synchronize camera events without recursive updates."""
    syncing = False

    def sync_camera(source, camera):
        nonlocal syncing
        if syncing:
            return
        camera_data = camera.to_plotly_json()
        syncing = True
        try:
            with figure.batch_update():
                for scene_name in scene_names:
                    target = figure.layout[scene_name]
                    if target.camera.to_plotly_json() != camera_data:
                        target.camera = camera_data
        finally:
            syncing = False

    for scene_name in scene_names:
        figure.layout[scene_name].on_change(sync_camera, "camera")


def make_interactive_pca_browser(
    pca_cache, *, color_by="position", width=1400, filter_position=False, midpoint=14.0,
    filter_displacement=False, min_displacement=2.0
):
    """Browse saved tracking/playback PCAs with one independent session slider.

    color_by selects starting-position, destination, or direction colors; all points are dots.
    The two panels' cameras are linked within this browser only. Creating two
    browsers gives independent session and camera controls without refitting.
    filter_position adds an all/left/right starting-position selector. Left is
    strictly below midpoint; right includes midpoint. Full-session axis ranges
    stay fixed when filtering, including when no points match.
    filter_displacement adds a minimum |final-initial| slider and excludes bouts
    with unknown endpoints. This filters saved coordinates without refitting.
    """
    if color_by not in ("position", "direction", "destination"):
        raise ValueError("color_by must be 'position', 'destination', or 'direction'.")
    # Use plain lists for widget data: NumPy typed-array acknowledgements can fail in Plotly.
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    session_info = pca_cache["session_info"]
    tracking_basis = pca_cache.get("projection_basis") == "tracking"
    slider = widgets.SelectionSlider(
        options=[
            (f"idx={idx} | {info['animal']} hs{info['headstage']} | session {info['session']}", idx)
            for idx, info in session_info.iterrows()
        ],
        description="Session", continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    position_filter = widgets.ToggleButtons(
        options=[("All", "all"), (f"Left (< {midpoint:g})", "left"), (f"Right (>= {midpoint:g})", "right")],
        description="Start:",
    ) if filter_position else None
    displacement_filter = widgets.FloatSlider(
        value=min_displacement, min=0,
        max=max(float(np.ceil(pca_cache["position_limits"][1] - pca_cache["position_limits"][0])), min_displacement),
        step=0.5, description="Min. |end-start|", readout_format=".1f",
        continuous_update=False, style={"description_width": "initial"},
        layout=widgets.Layout(width="95%"),
    ) if filter_displacement else None
    position_column = "final_position" if color_by == "destination" else "initial_position"
    encoding = {"position": "Starting position", "destination": "Destination position", "direction": "Direction"}[color_by]
    panels = ((0, "Tracking", encoding), (1, "Playback", encoding))
    figure = go.FigureWidget(make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[f"{name}: {encoding}" for _, name, encoding in panels],
        horizontal_spacing=0.02,
    ))
    figure._config = {"scrollZoom": True, "displaylogo": False}
    directions = ((-1, "Left", "#0072B2"), (1, "Right", "#D62728"), (0, "Zero / unknown direction", "#999999"))
    panel_traces = []
    for panel, (_, name, encoding) in enumerate(panels):
        traces = []
        if color_by == "destination":
            groups = ((None, encoding, None), ("missing", "Unknown destination", "#999999"))
        else:
            groups = ((None, encoding, None),) if color_by == "position" else directions
        for direction, label, color in groups:
            marker = {"symbol": "circle", "size": 4, "opacity": 0.7}
            marker.update({"coloraxis": "coloraxis"} if color is None else {"color": color})
            figure.add_trace(go.Scatter3d(
                x=[], y=[], z=[], mode="markers", name=label, legendgroup=label,
                marker=marker, showlegend=False,
                hovertemplate=(
                    "Bout %{customdata[0]:.0f}<br>Bin start: %{customdata[1]:.3f} s"
                    "<br>Starting position: %{customdata[2]:.2f}<br>Destination: %{customdata[4]:.2f}<br>Direction: %{customdata[3]}"
                    "<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
                ),
            ), row=panel // 2 + 1, col=panel % 2 + 1)
            traces.append((figure.data[-1], direction, label))
        panel_traces.append(traces)

    figure.update_layout(
        template="plotly_white", width=width, height=700,
        margin={"l": 0, "r": 65, "t": 85, "b": 40},
        coloraxis={
            "colorscale": "Viridis", "cmin": pca_cache["position_limits"][0],
            "cmax": pca_cache["position_limits"][1],
            "colorbar": {"title": encoding, "len": 0.85, "y": 0.5, "thickness": 15},
        },
        legend={"orientation": "h", "y": -0.02}, uirevision="preserve-pca-camera",
    )
    figure.update_scenes(aspectmode="data", camera={"eye": {"x": 1.5, "y": 1.5, "z": 1.2}})
    scene_names = ("scene", "scene2")
    _link_pca_cameras(figure, scene_names)

    def show_session(idx):
        info = session_info.loc[idx]
        fits = pca_cache["sessions"][idx]
        shown_legend_labels = set()
        with figure.batch_update():
            figure.layout.title.text = (
                f"idx={idx}: {info['animal']} hs{info['headstage']}, session {info['session']}"
                + ("<br><sup>Tracking PCA axes for both conditions; percentages describe tracking variance.</sup>"
                 if tracking_basis else "<br><sup>Drag to rotate; scroll to zoom. The two camera views are linked.</sup>")
            )
            for panel, (condition, name, encoding) in enumerate(panels):
                fit = fits.get(condition)
                available = fit is not None and fit["model"].n_components_ >= 3
                rows = fit["results"] if available else None
                if available and position_filter is not None:
                    if position_filter.value == "left":
                        rows = rows.loc[rows["initial_position"] < midpoint]
                    elif position_filter.value == "right":
                        rows = rows.loc[rows["initial_position"] >= midpoint]
                if available and displacement_filter is not None:
                    displacement = (rows["final_position"] - rows["initial_position"]).abs()
                    rows = rows.loc[displacement >= displacement_filter.value]
                figure.layout.annotations[panel].text = (
                    f"{name}: {encoding.lower()} ({len(rows)} / {len(fit['results'])} observations)" if available
                    else f"{name}: fewer than three components available"
                )
                scene = figure.layout[scene_names[panel]]
                for axis, pc in zip((scene.xaxis, scene.yaxis, scene.zaxis), range(3)):
                    axis.title.text = (
                        f"PC{pc + 1} ({fit['model'].explained_variance_ratio_[pc]:.1%})"
                        if available else f"PC{pc + 1}"
                    )
                    if available and (position_filter is not None or displacement_filter is not None or tracking_basis):
                        values = (
                            pd.concat([entry["results"][f"PC{pc + 1}"] for entry in fits.values()])
                            if tracking_basis else fit["results"][f"PC{pc + 1}"]
                        )
                        padding = max((values.max() - values.min()) * 0.05, 1e-6)
                        axis.range = [values.min() - padding, values.max() + padding]
                for trace, direction, label in panel_traces[panel]:
                    if not available:
                        trace.x, trace.y, trace.z, trace.customdata = [], [], [], []
                        trace.showlegend = False
                        continue
                    if color_by == "destination":
                        known = rows["final_position"].notna()
                        subset = rows.loc[known if direction is None else ~known]
                    elif direction is None:
                        subset = rows
                    elif direction == 0:
                        subset = rows.loc[~rows["initial_direction"].isin((-1, 1))]
                    else:
                        subset = rows.loc[rows["initial_direction"] == direction]
                    trace.x, trace.y, trace.z = (subset[pc].tolist() for pc in ("PC1", "PC2", "PC3"))
                    if direction is None:
                        trace.marker.color = subset[position_column].tolist()
                    customdata = np.empty((len(subset), 5), dtype=object)
                    customdata[:, 0] = subset.index.get_level_values("bout_id")
                    customdata[:, 1] = subset.index.get_level_values("bin_start_s")
                    customdata[:, 2] = subset["initial_position"]
                    customdata[:, 3] = subset["initial_direction"].map({-1: "Left", 1: "Right"}).fillna("Zero / unknown")
                    customdata[:, 4] = subset["final_position"].astype(object).where(subset["final_position"].notna(), None)
                    trace.customdata = customdata.tolist()
                    trace.showlegend = direction is not None and len(subset) > 0 and label not in shown_legend_labels
                    if trace.showlegend:
                        shown_legend_labels.add(label)

    slider.observe(lambda change: show_session(change["new"]), names="value")
    if position_filter is not None:
        position_filter.observe(lambda change: show_session(slider.value), names="value")
    if displacement_filter is not None:
        displacement_filter.observe(lambda change: show_session(slider.value), names="value")
    show_session(slider.value)
    controls = [slider]
    if position_filter is not None:
        controls.append(position_filter)
    if displacement_filter is not None:
        controls.append(displacement_filter)
    controls.append(figure)
    # VBox skips the figure's display hook; publish its populated initial state.
    figure._repr_mimebundle_()
    return widgets.VBox(controls, layout=widgets.Layout(width="100%", overflow="auto"))


def make_bout_pca_browser(pca_cache, *, width=1400):
    """Highlight one bout's chronological trajectory among saved PCA scores.

    A bout belongs to one condition, so it is highlighted only in that panel.
    Other observations are faint and colored by starting position. Session and bout controls update fixed traces;
    linked cameras and full-session axis ranges persist across bout selections.
    """
    # Use plain lists for widget data: NumPy typed-array acknowledgements can fail in Plotly.
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    session_info = pca_cache["session_info"]
    session_slider = widgets.SelectionSlider(
        options=[
            (f"idx={idx} | {info['animal']} hs{info['headstage']} | session {info['session']}", idx)
            for idx, info in session_info.iterrows()
        ],
        description="Session", continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    bout_selector = widgets.SelectionSlider(
        options=[("Loading bouts", -1)], description="Bout", continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    figure = go.FigureWidget(make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Tracking", "Playback"], horizontal_spacing=0.02,
    ))
    figure._config = {"scrollZoom": True, "displaylogo": False}
    for col in (1, 2):
        figure.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="markers", showlegend=False,
            marker={"size": 3, "coloraxis": "coloraxis2", "opacity": 0.22},
            hoverinfo="skip",
        ), row=1, col=col)
        figure.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines+markers+text", showlegend=False,
            line={"color": "#52677c", "width": 5},
            marker={"size": 8, "opacity": 1, "coloraxis": "coloraxis"},
            textposition="top center", textfont={"size": 14},
            hovertemplate=(
                "Bout %{customdata[0]:.0f}<br>Bin %{text}: %{customdata[1]:.3f} to %{customdata[2]:.3f} s"
                "<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
            ),
        ), row=1, col=col)
    figure.update_layout(
        template="plotly_white", width=width, height=750,
        margin={"l": 0, "r": 75, "t": 85, "b": 95},
        coloraxis={
            "colorscale": [[0, "#7db5e8"], [1, "#08306b"]],
            "colorbar": {"title": "Bin start (s)", "len": 0.85, "thickness": 15},
        },
        coloraxis2={
            "colorscale": "Viridis", "cmin": pca_cache["position_limits"][0],
            "cmax": pca_cache["position_limits"][1],
            "colorbar": {
                "title": "Starting position", "orientation": "h", "len": 0.7,
                "x": 0.5, "xanchor": "center", "y": -0.08, "yanchor": "top", "thickness": 12,
            },
        },
        uirevision="preserve-bout-camera",
    )
    figure.update_scenes(aspectmode="data", camera={"eye": {"x": 1.5, "y": 1.5, "z": 1.2}})
    _link_pca_cameras(figure)
    scene_names = ("scene", "scene2")
    current_fits = {}
    updating_options = False

    def highlight_bout():
        bout_id = bout_selector.value
        with figure.batch_update():
            for panel, (condition, name) in enumerate(((0, "Tracking"), (1, "Playback"))):
                fit = current_fits.get(condition)
                available = fit is not None and fit["model"].n_components_ >= 3
                background, highlighted = figure.data[panel * 2:panel * 2 + 2]
                if not available:
                    background.x, background.y, background.z, background.marker.color = [], [], [], []
                    highlighted.x, highlighted.y, highlighted.z = [], [], []
                    highlighted.text, highlighted.customdata, highlighted.marker.color = [], [], []
                    figure.layout.annotations[panel].text = f"{name}: fewer than three components available"
                    continue
                rows = fit["results"]
                selected = rows.loc[rows.index.get_level_values("bout_id") == bout_id].sort_values("bin_number")
                other = rows.loc[rows.index.get_level_values("bout_id") != bout_id]
                for trace, subset in ((background, other), (highlighted, selected)):
                    trace.x, trace.y, trace.z = (subset[pc].tolist() for pc in ("PC1", "PC2", "PC3"))
                background.marker.color = other["initial_position"].tolist()
                highlighted.text = selected["bin_number"].astype(str).tolist()
                highlighted.marker.color = selected.index.get_level_values("bin_start_s").tolist()
                highlighted.customdata = np.column_stack((
                    selected.index.get_level_values("bout_id"),
                    selected.index.get_level_values("bin_start_s"), selected["bin_end_s"],
                )).tolist()
                figure.layout.annotations[panel].text = (
                    f"{name}: bout {bout_id} ({len(selected)} windows)" if len(selected)
                    else f"{name}: selected bout is in the other condition"
                )

    def show_session(idx):
        nonlocal current_fits, updating_options
        current_fits = pca_cache["sessions"][idx]
        info = session_info.loc[idx]
        bout_options = []
        with figure.batch_update():
            figure.layout.title.text = (
                f"idx={idx}: {info['animal']} hs{info['headstage']}, session {info['session']}"
                "<br><sup>Selected bout: light to dark, connected in time order; other bouts colored by starting position.</sup>"
            )
            for panel, (condition, name) in enumerate(((0, "Tracking"), (1, "Playback"))):
                fit = current_fits.get(condition)
                if fit is None or fit["model"].n_components_ < 3:
                    continue
                rows = fit["results"]
                scene = figure.layout[scene_names[panel]]
                for pc, axis in enumerate((scene.xaxis, scene.yaxis, scene.zaxis)):
                    values = rows[f"PC{pc + 1}"]
                    padding = max((values.max() - values.min()) * 0.05, 1e-6)
                    axis.range = [values.min() - padding, values.max() + padding]
                    axis.title.text = f"PC{pc + 1} ({fit['model'].explained_variance_ratio_[pc]:.1%})"
                for bout_id in rows.index.get_level_values("bout_id").unique():
                    bout_options.append((f"Bout {bout_id} | {name}", bout_id))
                starts = rows.index.get_level_values("bin_start_s")
                figure.layout.coloraxis.cmin = starts.min()
                figure.layout.coloraxis.cmax = starts.max() if starts.max() > starts.min() else rows["bin_end_s"].max()
            updating_options = True
            try:
                bout_selector.options = bout_options or [("No bouts available", -1)]
                bout_selector.disabled = not bout_options
                bout_selector.value = bout_options[0][1] if bout_options else -1
            finally:
                updating_options = False
            highlight_bout()

    def on_bout_change(change):
        if not updating_options:
            highlight_bout()

    session_slider.observe(lambda change: show_session(change["new"]), names="value")
    bout_selector.observe(on_bout_change, names="value")
    show_session(session_slider.value)
    # VBox skips the figure's display hook; publish its populated initial state.
    figure._repr_mimebundle_()
    return widgets.VBox(
        [session_slider, bout_selector, figure],
        layout=widgets.Layout(width="100%", overflow="auto"),
    )


def make_start_end_position_browser(movement_bouts, session_info, *, width=1500):
    """Compare bout endpoints, counting each reference-table row once.

    Per session, show tracking/playback start-versus-end plots and empirical
    cumulative distributions of absolute net displacement. Unknown endpoints
    are excluded, and the count is displayed. No PCA scores are involved.
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    slider = widgets.SelectionSlider(
        options=[
            (f"idx={idx} | {info['animal']} hs{info['headstage']} | session {info['session']}", idx)
            for idx, info in session_info.iterrows()
        ],
        description="Session", continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    positions = movement_bouts[["initial_position", "final_position"]]
    lower, upper = float(positions.min().min()), float(positions.max().max())
    figure = go.FigureWidget(make_subplots(
        rows=1, cols=3, subplot_titles=["Tracking", "Playback", "How close is the endpoint?"],
        horizontal_spacing=0.08,
    ))
    for col, (name, color) in enumerate((("Tracking", "red"), ("Playback", "black")), 1):
        figure.add_trace(go.Scattergl(
            x=[], y=[], mode="markers", name=name, showlegend=False,
            marker={"color": color, "size": 6, "opacity": 0.4},
            hovertemplate="Bout %{customdata}<br>Start: %{x:.3f}<br>End: %{y:.3f}<extra></extra>",
        ), row=1, col=col)
        figure.add_shape(
            type="line", x0=lower, y0=lower, x1=upper, y1=upper,
            line={"color": "gray", "dash": "dash", "width": 1}, row=1, col=col,
        )
        figure.update_xaxes(title_text="Starting position", range=[lower, upper], row=1, col=col)
        figure.update_yaxes(
            title_text="Final position", range=[lower, upper],
            scaleanchor="x" if col == 1 else "x2", scaleratio=1, row=1, col=col,
        )
    for name, color in (("Tracking", "red"), ("Playback", "black")):
        figure.add_trace(go.Scatter(
            x=[], y=[], mode="lines", name=name,
            line={"color": color, "shape": "hv"},
            hovertemplate="Displacement <= %{x:.3f}<br>Fraction of bouts: %{y:.1%}<extra>%{fullData.name}</extra>",
        ), row=1, col=3)
    figure.update_xaxes(title_text="Absolute net displacement |end - start|", range=[0, upper - lower], row=1, col=3)
    figure.update_yaxes(title_text="Fraction of bouts at or below this distance", range=[0, 1], row=1, col=3)
    figure.update_layout(
        template="plotly_white", width=width, height=540,
        margin={"l": 50, "r": 20, "t": 120, "b": 75},
        legend={"orientation": "h", "y": -0.25},
    )

    def show_session(idx):
        info = session_info.loc[idx]
        bouts = movement_bouts.loc[movement_bouts["idx"] == idx]
        complete = bouts.dropna(subset=["initial_position", "final_position"])
        with figure.batch_update():
            figure.layout.title.text = (
                f"idx={idx}: {info['animal']} hs{info['headstage']}, session {info['session']}"
                f"<br><sup>One point per bout. Dashed line: end = start. Unknown endpoints excluded: {len(bouts) - len(complete)}.</sup>"
            )
            for panel, (condition, name) in enumerate(((0, "Tracking"), (1, "Playback"))):
                rows = complete.loc[complete["condition"] == condition]
                displacement = (rows["final_position"] - rows["initial_position"]).abs()
                r = rows["initial_position"].corr(rows["final_position"]) if len(rows) > 1 else np.nan
                points = figure.data[panel]
                points.x, points.y = rows["initial_position"].tolist(), rows["final_position"].tolist()
                points.customdata = rows.index.tolist()
                figure.layout.annotations[panel].text = (
                    f"{name}: {len(rows)} bouts<br>r = {r:.2f}; median |end-start| = {displacement.median():.2f}"
                )
                cdf = figure.data[panel + 2]
                cdf.x = [0.0] + np.sort(displacement.to_numpy()).tolist()
                cdf.y = [0.0] + (np.arange(1, len(rows) + 1) / len(rows)).tolist() if len(rows) else [0.0]

    slider.observe(lambda change: show_session(change["new"]), names="value")
    show_session(slider.value)
    # VBox skips the figure's display hook; publish its populated initial state.
    figure._repr_mimebundle_()
    return widgets.VBox([slider, figure], layout=widgets.Layout(width="100%", overflow="auto"))
