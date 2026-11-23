"""Onset-aligned PC time courses projected onto saved tracking PCA bases."""

import numpy as np
import pandas as pd

from psth import extract_neuron_counts, prepare_psth_rates


def project_pc_psths(neural_data, neuron_ids, bouts, mode_fits, *, dt,
                     window_s=(-1.2, 1.0), bin_width_s=.005,
                     baseline_window_s=(-1., -.7), smoothing_sigma_s=.01,
                     n_components=3):
    """Project each bout using fixed saved tracking loadings AND mean.

    mode_fits maps raw/baseline_subtracted to saved tracking PCA models. Neuron
    identity/order must match exactly. Subtract neural baselines before temporal
    smoothing and projection; no per-bout PC re-zeroing, scaling or PCA fitting.
    Complete-window selection is identical for both modes and all neurons.
    """
    if set(mode_fits) != {"raw", "baseline_subtracted"}:
        raise ValueError("Supply saved tracking PCA models for raw and baseline_subtracted.")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if neural_data.ndim != 2 or neural_data.shape[0] != len(neuron_ids) or not len(neuron_ids):
        raise ValueError("Recording neuron axis must match the saved neuron IDs.")
    if not pd.Index(neuron_ids).is_unique:
        raise ValueError("Recording neuron IDs must be unique.")
    for model in mode_fits.values():
        if not hasattr(model, "feature_names_in_") or list(model.feature_names_in_) != list(neuron_ids):
            raise ValueError("Tracking PCA and recording neuron IDs/order must match exactly.")
        if model.n_components_ < n_components or model.whiten:
            raise ValueError("Need sufficient components from an unwhitened PCA.")
    models = list(mode_fits.values())
    if any(model.n_features_in_ != len(neuron_ids) for model in models):
        raise ValueError("PCA dimensions do not match recording neurons.")
    counts, labels, excluded = extract_neuron_counts(
        neural_data, bouts, neuron=0, dt=dt, window_s=window_s,
    )
    result = {"labels": labels, "excluded_bouts": excluded, "modes": {},
              "neuron_ids": list(neuron_ids), "settings": {
                  "dt": dt, "window_s": list(window_s), "bin_width_s": bin_width_s,
                  "baseline_window_s": list(baseline_window_s),
                  "smoothing_sigma_s": smoothing_sigma_s,
                  "n_components": n_components, "projection_basis": "tracking",
                  "score_baseline_rezeroed": False}}
    if labels.empty:
        raise ValueError("No complete bout windows are available.")
    for mode, model in mode_fits.items():
        # Linearity permits accumulating weighted neural rates without copying a
        # complete bout x neuron x time array or re-estimating any PCA mean.
        scores = None
        for neuron in range(len(neuron_ids)):
            if neuron:
                counts, neuron_labels, _ = extract_neuron_counts(
                    neural_data, bouts, neuron=neuron, dt=dt, window_s=window_s,
                )
                if not neuron_labels.index.equals(labels.index):
                    raise ValueError("Neurons have inconsistent bout selection.")
            else:
                counts, _, _ = extract_neuron_counts(
                    neural_data, bouts, neuron=0, dt=dt, window_s=window_s,
                )
            rates, centers = prepare_psth_rates(
                counts, dt=dt, window_s=window_s, bin_width_s=bin_width_s,
                baseline_window_s=baseline_window_s if mode == "baseline_subtracted" else None,
                smoothing_sigma_s=smoothing_sigma_s,
            )
            if not np.isfinite(rates).all():
                raise ValueError("Nonfinite neural rates in the selected bout windows.")
            contribution = (rates - model.mean_[neuron])[:, :, None] * model.components_[:n_components, neuron]
            scores = contribution if scores is None else scores + contribution
        result["modes"][mode] = {"model": model, "conditions": {}}
        for condition in (0, 1):
            selected = labels["condition"].eq(condition).to_numpy()
            traces = scores[selected]
            n = len(traces)
            result["modes"][mode]["conditions"][condition] = {
                "bout_ids": labels.index[selected].tolist(), "bout_scores": traces,
                "n_bouts": n,
                "mean": traces.mean(axis=0) if n else np.full(scores.shape[1:], np.nan),
                "sem": traces.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(scores.shape[1:], np.nan),
            }
    result["centers_s"] = centers
    return result


def summarize_pc_psths(result):
    """Reduce one projected session to all/left/right display summaries.

    Retain mean/SEM, counts, bout IDs, and references to the fixed models, but no
    per-bout score arrays or source recordings. Directions are selected after
    projection, which gives exactly the same traces as projecting each subset.
    Unknown directions enter only the all-directions summary.
    """
    summaries = {}
    for direction in (None, -1, 1):
        labels = result["labels"]
        excluded = result["excluded_bouts"]
        selected_labels = labels if direction is None else labels.loc[labels["initial_direction"].eq(direction)]
        selected_excluded = excluded if direction is None else excluded.loc[excluded["initial_direction"].eq(direction)]
        summary = {"settings": {**result["settings"], "direction": direction},
                   "centers_s": result["centers_s"], "neuron_ids": result["neuron_ids"],
                   "n_complete_bouts": len(selected_labels), "n_excluded_bouts": len(selected_excluded),
                   "modes": {}}
        if "session_info" in result:
            summary["session_info"] = result["session_info"]
        for mode, values in result["modes"].items():
            summary["modes"][mode] = {"model": values["model"], "conditions": {}}
            for condition, original in values["conditions"].items():
                ids = pd.Index(original["bout_ids"])
                keep = ids.isin(selected_labels.index)
                traces = original["bout_scores"][keep]
                n = len(traces)
                shape = original["mean"].shape
                summary["modes"][mode]["conditions"][condition] = {
                    "bout_ids": ids[keep].tolist(), "n_bouts": n,
                    "mean": traces.mean(axis=0) if n else np.full(shape, np.nan),
                    "sem": traces.std(axis=0, ddof=1)/np.sqrt(n) if n > 1 else np.full(shape, np.nan),
                }
        summaries[direction] = summary
    return summaries


def plot_pc_psths(result, *, display_window_s=(-.5, .5), title=""):
    """Draw PC time courses with one signed loading bar per recorded unit."""
    import matplotlib.pyplot as plt

    centers = result["centers_s"]
    if not (result["settings"]["window_s"][0] <= display_window_s[0] < display_window_s[1] <= result["settings"]["window_s"][1]):
        raise ValueError("Display window must be contained in extraction.")
    visible = (centers >= display_window_s[0]) & (centers < display_window_s[1])
    if not visible.any():
        raise ValueError("No displayed time bins.")
    n_components = result["settings"]["n_components"]
    fig = plt.figure(figsize=(5*n_components, 11), layout="constrained")
    grid = fig.add_gridspec(4, n_components, height_ratios=[3, 1, 3, 1])
    axes = np.empty((2, n_components), dtype=object)
    loadings = np.concatenate([result["modes"][mode]["model"].components_[:n_components]
                               for mode in ("raw", "baseline_subtracted")])
    limit = max(float(np.abs(loadings).max()) * 1.05, .01)
    unit_positions = np.arange(len(result["neuron_ids"]))
    unit_labels = [str(unit).rsplit("_neuron", 1)[-1] for unit in result["neuron_ids"]]
    for row, mode in enumerate(("raw", "baseline_subtracted")):
        model = result["modes"][mode]["model"]
        for pc in range(n_components):
            ax = fig.add_subplot(grid[2*row, pc], sharex=axes[0, 0] if row or pc else None)
            axes[row, pc] = ax
            weight_ax = fig.add_subplot(grid[2*row+1, pc])
            weights = model.components_[pc]
            weight_ax.bar(unit_positions, weights, color=".55", width=.8)
            weight_ax.axhline(0, color=".25", linewidth=.8)
            weight_ax.set(xlabel="Unit", ylabel="PC weight",
                          xlim=(-.7, len(weights)-.3), ylim=(-limit, limit))
            weight_ax.set_xticks(unit_positions, unit_labels)
            weight_ax.tick_params(axis="x", labelsize=7,
                                  rotation=45 if any(len(label) > 3 for label in unit_labels) else 0)
            weight_ax.tick_params(axis="y", labelsize=8)
            weight_ax.text(.98, .92, f"n={len(weights)} units", transform=weight_ax.transAxes,
                           ha="right", va="top", fontsize=8)
            for condition, name, color in ((0, "Tracking", "red"), (1, "Playback", "black")):
                values = result["modes"][mode]["conditions"][condition]
                mean, sem = values["mean"][visible, pc], values["sem"][visible, pc]
                ax.plot(centers[visible], mean, color=color, label=f"{name}: {values['n_bouts']} bouts")
                if values["n_bouts"] > 1:
                    ax.fill_between(centers[visible], mean-sem, mean+sem, color=color, alpha=.18)
            ax.axvline(0, color=".5", linestyle="--", linewidth=1)
            ax.axhline(0, color=".65", linewidth=.7)
            percent = model.explained_variance_ratio_[pc]*100
            mode_name = "Raw neural activity" if mode == "raw" else "Baseline-subtracted neural activity"
            ax.set(title=f"{mode_name}\nTracking PC{pc+1} ({percent:.1f}% fit variance)",
                   xlabel="Time from movement onset (s)", ylabel="PC score (weighted spikes/s)",
                   xlim=display_window_s)
            if pc == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}\nShared tracking axes within each row; shading = SEM across bouts")
    return fig, axes
