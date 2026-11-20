"""Perimotor counts and descriptive PSTHs aligned to saved movement onsets."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def extract_neuron_counts(neural_data, bouts, *, neuron, dt, window_s):
    """Return complete bout windows, aligned labels, and excluded bout labels.

    Bin bounds are start-inclusive/end-exclusive. Original bout IDs are preserved.
    Only the selected neuron's windows are copied from the full recording.
    """
    edges = np.asarray(window_s, dtype=float)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")
    if edges.shape != (2,) or not np.isfinite(edges).all() or edges[0] >= edges[1]:
        raise ValueError("window_s must contain finite increasing bounds.")
    offsets = np.rint(edges / dt).astype(int)
    if not np.allclose(offsets * dt, edges, rtol=0, atol=1e-10):
        raise ValueError("Window bounds must align to original bins.")
    if neural_data.ndim != 2 or not 0 <= neuron < neural_data.shape[0]:
        raise ValueError("Select a neuron from a neurons-by-time recording.")
    if not bouts.index.is_unique:
        raise ValueError("Bout IDs must be unique.")
    onsets = bouts["onset_bin"].to_numpy(dtype=int)
    starts, stops = onsets + offsets[0], onsets + offsets[1]
    valid = ((starts >= 0) & (stops <= neural_data.shape[1])
             & (onsets >= 0) & (onsets < neural_data.shape[1]))
    counts = np.empty((valid.sum(), offsets[1] - offsets[0]), dtype=float)
    for row, (start, stop) in enumerate(zip(starts[valid], stops[valid])):
        counts[row] = neural_data[neuron, start:stop]
    return counts, bouts.loc[valid].copy(), bouts.loc[~valid].copy()


def bin_firing_rates(counts, *, dt, bin_width_s):
    """Sum counts in nonoverlapping bins and convert each bout to spikes/s."""
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(bin_width_s) or bin_width_s <= 0:
        raise ValueError("Bin durations must be positive and finite.")
    factor = round(bin_width_s / dt)
    if factor < 1 or not np.isclose(factor * dt, bin_width_s, rtol=0, atol=1e-10):
        raise ValueError("PSTH bin width must be a multiple of the original bin width.")
    if counts.ndim != 2 or counts.shape[1] % factor:
        raise ValueError("PSTH bin width must divide the complete window.")
    return counts.reshape(counts.shape[0], counts.shape[1] // factor, factor).sum(-1) / bin_width_s


def prepare_psth_rates(counts, *, dt, window_s, bin_width_s,
                       baseline_window_s=None, smoothing_sigma_s=0):
    """Subtract each bout's raw baseline rate, then optionally smooth in time.

    Smoothing uses a symmetric Gaussian (truncate=4, nearest edge padding).
    Apply it to the full extraction before limiting the displayed interval.
    """
    if not np.isclose(counts.shape[1] * dt, window_s[1] - window_s[0]):
        raise ValueError("Counts do not match the requested window.")
    rates = bin_firing_rates(counts, dt=dt, bin_width_s=bin_width_s)
    if baseline_window_s is not None:
        bounds = np.asarray(baseline_window_s, dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1] or bounds[0] < window_s[0]
                or bounds[1] > min(0, window_s[1])):
            raise ValueError("Baseline must be a pre-onset interval inside the extraction.")
        offsets = (bounds - window_s[0]) / dt
        bins = np.rint(offsets).astype(int)
        if not np.allclose(bins, offsets, rtol=0, atol=1e-8):
            raise ValueError("Baseline bounds must align to original bins.")
        baseline_rates = counts[:, bins[0]:bins[1]].sum(axis=1) / ((bins[1] - bins[0]) * dt)
        rates = rates - baseline_rates[:, None]
    if not np.isfinite(smoothing_sigma_s) or smoothing_sigma_s < 0:
        raise ValueError("Smoothing sigma must be finite and nonnegative.")
    if smoothing_sigma_s > 0 and len(rates):
        rates = gaussian_filter1d(rates, smoothing_sigma_s / bin_width_s,
                                  axis=1, mode="nearest", truncate=4)
    centers = window_s[0] + (np.arange(rates.shape[1]) + 0.5) * bin_width_s
    return rates, centers


def session_neuron_psths(neural_data, bouts, *, dt, window_s, bin_width_s,
                         baseline_window_s=None, smoothing_sigma_s=0):
    """Return one bout-averaged PSTH per neuron and condition for a session.

    Outputs are condition-keyed arrays (neurons, time bins), time centers, and
    complete bout counts per condition. Missing conditions have zero neuron rows.
    Keep neuron rows until the final population mean so each neuron has equal weight.
    """
    traces = {0: [], 1: []}
    n_bouts = {0: 0, 1: 0}
    centers = None
    for neuron in range(neural_data.shape[0]):
        counts, labels, _ = extract_neuron_counts(
            neural_data, bouts, neuron=neuron, dt=dt, window_s=window_s,
        )
        rates, centers = prepare_psth_rates(
            counts, dt=dt, window_s=window_s, bin_width_s=bin_width_s,
            baseline_window_s=baseline_window_s, smoothing_sigma_s=smoothing_sigma_s,
        )
        for condition in (0, 1):
            selected = labels["condition"].eq(condition).to_numpy()
            n_bouts[condition] = int(selected.sum())
            if n_bouts[condition]:
                traces[condition].append(rates[selected].mean(axis=0))
    if centers is None:
        raise ValueError("Recording contains no neurons.")
    return {
        condition: np.stack(rows) if rows else np.empty((0, len(centers)))
        for condition, rows in traces.items()
    }, centers, n_bouts


def plot_population_psth(neuron_psths, centers, *, display_window_s, title,
                         baseline_subtracted=True, smoothed=True):
    """Average neuron PSTHs equally; shaded SEM is across neuron PSTHs."""
    visible = (centers >= display_window_s[0]) & (centers < display_window_s[1])
    if not visible.any():
        raise ValueError("Display window contains no PSTH bins.")
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    for condition, name, color in [(0, "Tracking", "red"), (1, "Playback", "black")]:
        values = np.asarray(neuron_psths[condition])
        if values.ndim != 2 or values.shape[1] != len(centers):
            raise ValueError("Expected neuron-by-time PSTHs aligned to time centers.")
        values = values[:, visible]
        n = len(values)
        if not n:
            ax.plot([], [], color=color, label=f"{name}: 0 neurons")
            continue
        mean = values.mean(axis=0)
        ax.plot(centers[visible], mean, color=color,
                drawstyle="default" if smoothed else "steps-mid",
                label=f"{name}: {n} neurons")
        if n > 1:
            sem = values.std(axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(centers[visible], mean - sem, mean + sem,
                            step=None if smoothed else "mid", color=color, alpha=.18)
    ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
    if baseline_subtracted:
        ax.axhline(0, color="0.5", linewidth=.8)
    ax.set(xlim=display_window_s, xlabel="Time from movement onset (s)",
           ylabel="Change in firing rate (spikes/s)" if baseline_subtracted else "Firing rate (spikes/s)",
           title=f"{title}\nEqual weight per neuron; shading = SEM across neurons")
    ax.legend()
    return fig, ax


def plot_neuron_psth(counts, labels, *, dt, window_s, bin_width_s, title,
                     display_window_s=None, baseline_window_s=None, smoothing_sigma_s=0):
    """Plot mean +/- SEM and both conditions' original-bin rasters.

    SEM is across bouts (not a confidence interval or a significance test).
    Raster marks denote occupied original time bins; size reflects spike count.
    """
    if len(counts) != len(labels):
        raise ValueError("Counts and labels must have the same row order and length.")
    rates, centers = prepare_psth_rates(
        counts, dt=dt, window_s=window_s, bin_width_s=bin_width_s,
        baseline_window_s=baseline_window_s, smoothing_sigma_s=smoothing_sigma_s,
    )
    if display_window_s is None:
        display_window_s = window_s
    if not (window_s[0] <= display_window_s[0] < display_window_s[1] <= window_s[1]):
        raise ValueError("Display window must fit within the extraction.")
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(grid[0, :])
    rasters = [fig.add_subplot(grid[1, i], sharex=ax) for i in range(2)]
    for condition, name, color, raster in zip(
        [0, 1], ["Tracking", "Playback"], ["red", "black"], rasters
    ):
        selected = labels["condition"].eq(condition).to_numpy()
        n = int(selected.sum())
        if n:
            mean = rates[selected].mean(axis=0)
            ax.plot(centers, mean, drawstyle="default" if smoothing_sigma_s else "steps-mid",
                    color=color, label=f"{name}: {n} bouts")
            if n > 1:
                sem = rates[selected].std(axis=0, ddof=1) / np.sqrt(n)
                ax.fill_between(centers, mean - sem, mean + sem,
                                step=None if smoothing_sigma_s else "mid", color=color, alpha=0.18)
            rows, bins = np.nonzero(counts[selected])
            raster.scatter(window_s[0] + (bins + 0.5) * dt, rows + 1,
                           s=5 * np.sqrt(counts[selected][rows, bins]),
                           marker="|", color=color, linewidths=0.6)
        else:
            ax.plot([], [], color=color, label=f"{name}: 0 bouts")
            raster.text(0.5, 0.5, "No complete bouts", transform=raster.transAxes, ha="center")
        raster.set(title=f"{name} raster ({n} bouts)", xlabel="Time from movement onset (s)",
                   ylabel="Bout (chronological order)", ylim=(max(n, 1) + 0.5, 0.5))
    for axis in [ax, *rasters]:
        axis.axvline(0, color="0.5", linestyle="--", linewidth=1)
        axis.set_xlim(display_window_s)
    # Keep hidden extraction/baseline activity out of the displayed y-axis scaling.
    visible = (centers >= display_window_s[0]) & (centers < display_window_s[1])
    heights = []
    for condition in [0, 1]:
        values = rates[labels["condition"].eq(condition).to_numpy()][:, visible]
        if len(values):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            heights.extend([np.min(mean - sem), np.max(mean + sem)])
    if heights:
        low, high = min(heights), max(heights)
        if baseline_window_s is not None:
            low, high = min(low, 0), max(high, 0)
        pad = max((high - low) * 0.08, 0.5)
        ax.set_ylim(low - pad, high + pad)
    if baseline_window_s is not None:
        ax.axhline(0, color="0.5", linewidth=0.8)
    baseline_text = (f"baseline [{baseline_window_s[0]:g}, {baseline_window_s[1]:g}) s"
                     if baseline_window_s is not None else "raw rate")
    smooth_text = f"Gaussian sigma={smoothing_sigma_s * 1000:g} ms" if smoothing_sigma_s else "unsmoothed"
    ax.set(ylabel="Change in firing rate (spikes/s)" if baseline_window_s is not None else "Firing rate (spikes/s)",
           xlabel="Time from movement onset (s)",
           title=f"{bin_width_s * 1000:g} ms bins | {smooth_text} | {baseline_text}\nmean +/- SEM across bouts")
    ax.legend(loc="upper left")
    fig.suptitle(title, fontsize=11)
    return fig, (ax, *rasters)
