"""Shared data/label helpers for representation evaluation.

See ANALYSIS_DATA_FORMAT.md. Metric functions belong here, separately from
extraction, model fitting and interactive plotting in utils.py.
"""

import numpy as np
import pandas as pd


def validate_data_labels(data, labels):
    """Check the observation format without changing either table.

    Numeric features/coordinates must be finite. Behavioral labels may contain
    missing values; each metric must handle missing values in its own target.
    Empty selections are valid tables but may be insufficient for a metric.
    """
    if not isinstance(data, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise TypeError("data and labels must both be pandas DataFrames.")
    if not isinstance(data.index, pd.MultiIndex) or data.index.names != ["bout_id", "bin_start_s"]:
        raise ValueError("Use a MultiIndex named (bout_id, bin_start_s).")
    if not isinstance(labels.index, pd.MultiIndex) or labels.index.names != data.index.names:
        raise ValueError("labels must use the same named observation index as data.")
    if not data.index.is_unique or not labels.index.is_unique:
        raise ValueError("Each bout/window observation must have a unique index.")
    if not data.index.equals(labels.index):
        raise ValueError("data and labels must have identical indices in identical row order.")
    if data.shape[1] == 0 or not data.columns.is_unique or not labels.columns.is_unique:
        raise ValueError("Use at least one data column and unique column names in each table.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError("data must contain only numeric features or coordinates.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("data contains missing or infinite values.")


def fit_label_regression(data, labels, *, target):
    """Describe a continuous label with ordinary linear regression and an intercept.

    data contains the explicitly selected neuronal features or coordinates;
    labels supplies target. Both use the same bout/window observation index.
    Exclude only rows with missing/nonfinite target values, preserving used IDs.
    Other labels do not affect selection. At least two valid observations and
    a nonconstant numeric target are required.

    Return the fitted model, target name, in-sample R2/MAE, adjusted R2,
    n_features, df_residual, n_excluded, and an indexed predictions table
    (observed, fitted, residual = observed - fitted). Degrees of freedom use
    the fitted design's rank, including the intercept. Adjusted R2 is NaN when
    no residual degrees of freedom remain.
    No scaling, dimensionality reduction, cross-validation, or randomization is
    performed. These are descriptive fit scores, not held-out performance.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    validate_data_labels(data, labels)
    y = labels[target]
    if not pd.api.types.is_numeric_dtype(y.dtype):
        raise ValueError(f"Target {target!r} must be numeric and interpreted as continuous.")
    y = y.astype(float)
    keep = np.isfinite(y)
    x, y = data.loc[keep], y.loc[keep]
    if len(y) < 2:
        raise ValueError("At least two observations with finite target values are required.")
    if y.nunique() < 2:
        raise ValueError(f"Target {target!r} has no variation in the selected observations.")

    model = LinearRegression().fit(x, y)
    fitted = model.predict(x)
    predictions = pd.DataFrame(
        {"observed": y, "fitted": fitted, "residual": y - fitted}, index=x.index
    )
    r2 = float(r2_score(y, fitted))
    df_residual = len(y) - int(model.rank_) - 1
    r2_adjusted = 1 - (1 - r2) * (len(y) - 1) / df_residual if df_residual > 0 else np.nan
    return {
        "model": model,
        "target": target,
        "r2": r2,
        "r2_adjusted": float(r2_adjusted),
        "n_features": x.shape[1],
        "df_residual": df_residual,
        "mae": float(mean_absolute_error(y, fitted)),
        "n_excluded": int((~keep).sum()),
        "predictions": predictions,
    }


def plot_label_regression(regression, *, ax=None, color="black"):
    """Plot observed versus fitted values from fit_label_regression(), without fitting."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    predictions = regression["predictions"]
    ax.scatter(predictions["observed"], predictions["fitted"], s=18, alpha=0.5, color=color)
    lower = float(predictions[["observed", "fitted"]].min().min())
    upper = float(predictions[["observed", "fitted"]].max().max())
    padding = max((upper - lower) * 0.05, 1e-6)
    limits = (lower - padding, upper + padding)
    ax.plot(limits, limits, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xlabel=f"Observed {regression['target']}", ylabel=f"Fitted {regression['target']}",
        xlim=limits, ylim=limits,
        title=(f"Descriptive R² = {regression['r2']:.3f} | MAE = {regression['mae']:.2f}"
               f"\nn = {len(predictions)}; excluded = {regression['n_excluded']}"),
    )
    ax.set_aspect("equal", adjustable="box")
    return ax.figure, ax


def run_label_regressions(pca_cache, *, target, columns, min_displacement=None):
    """Fit one pooled regression per session and condition, using all windows.

    Select columns directly from each saved results table. An optional strict
    abs(final-initial) displacement filter keeps all windows of qualifying bouts.
    Invalid/insufficient subsets get a skipped status/reason. Return summary,
    fits and settings; summary/fit keys are (idx, condition). Prediction tables
    retain the full (bout_id, bin_start_s) index.

    Report both n_observations (bout/window rows) and n_bouts. Degrees of freedom
    and adjusted R2 use the design's row count; they do not correct for dependence
    among windows from the same bout. These are descriptive pooled fits, with no
    PCA refitting, held-out scoring, resampling or file I/O.
    """
    if isinstance(columns, str):
        raise TypeError("columns must be a sequence of feature names.")
    columns = list(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Supply a nonempty sequence of unique feature names.")
    if min_displacement is not None and (
        not np.isfinite(min_displacement) or min_displacement < 0
    ):
        raise ValueError("min_displacement must be None or a finite nonnegative number.")

    records, fits = [], {}
    for idx, condition_fits in pca_cache["sessions"].items():
        info = pca_cache["session_info"].loc[idx]
        for condition, pca_fit in condition_fits.items():
            rows = pca_fit["results"]
            record = {
                "idx": idx, "condition": condition,
                "n_windows": int(rows["bin_number"].nunique()),
                "animal": info["animal"], "headstage": info["headstage"],
                "session": info["session"], "region": info.get("region", None),
                "time_start_s": float(rows.index.get_level_values("bin_start_s").min()),
                "time_end_s": float(rows["bin_end_s"].max()),
                "target": target, "n_features": len(columns),
                "n_candidates": len(rows), "n_filtered": 0,
                "n_excluded_target": 0, "n_observations": 0, "n_bouts": 0,
                "r2": np.nan, "r2_adjusted": np.nan, "mae": np.nan,
                "df_residual": np.nan, "status": "skipped", "reason": "",
            }
            required = set(columns) | {target}
            if min_displacement is not None:
                required.update(("initial_position", "final_position"))
            missing = required.difference(rows.columns)
            if missing:
                record["reason"] = "Missing columns: " + ", ".join(sorted(missing))
                records.append(record)
                continue

            selected = rows
            if min_displacement is not None:
                displacement = (rows["final_position"] - rows["initial_position"]).abs()
                selected = rows.loc[np.isfinite(displacement) & (displacement > min_displacement)]
            record["n_filtered"] = len(rows) - len(selected)
            if pd.api.types.is_numeric_dtype(selected[target].dtype):
                valid_target = np.isfinite(selected[target].astype(float))
                record["n_excluded_target"] = int((~valid_target).sum())
                record["n_observations"] = int(valid_target.sum())
                record["n_bouts"] = selected.index.get_level_values("bout_id")[valid_target].nunique()

            try:
                fit = fit_label_regression(selected[columns], selected, target=target)
            except ValueError as error:
                record["reason"] = str(error)
            else:
                key = (idx, condition)
                fits[key] = fit
                record.update({name: fit[name] for name in ("r2", "r2_adjusted", "mae", "df_residual")})
                record["status"] = "ok"
            records.append(record)

    if not records:
        raise ValueError("No session/condition subsets are available to evaluate.")
    summary = pd.DataFrame(records).set_index(["idx", "condition"])
    return {
        "summary": summary,
        "fits": fits,
        "settings": {
            "target": target, "columns": columns, "min_displacement": min_displacement,
            "displacement_rule": "abs(final_position - initial_position) > cutoff",
            "window_mode": "pooled", "scoring": "descriptive_in_sample",
            "projection_basis": pca_cache.get("projection_basis", "separate_condition_pcas"),
            "pca_settings": dict(pca_cache.get("settings", {})),
        },
    }


def plot_regression_summary(summary, *, score="r2_adjusted"):
    """Plot one score per session/condition PCA, without time-window splitting."""
    import matplotlib.pyplot as plt

    if summary.index.names != ["idx", "condition"] or not summary.index.is_unique:
        raise ValueError("Use the pooled summary indexed by (idx, condition).")
    rows = summary.reset_index()
    rows = rows.loc[(rows["status"] == "ok") & np.isfinite(rows[score])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    ylabel = {"r2": "Descriptive R-squared", "r2_adjusted": "Adjusted R-squared", "mae": "MAE"}.get(score, score)
    for ax, (condition, name, color) in zip(axes, ((0, "Tracking", "red"), (1, "Playback", "black"))):
        condition_rows = rows.loc[rows["condition"] == condition].sort_values("idx")
        if len(condition_rows):
            ax.scatter(condition_rows["idx"], condition_rows[score], color=color, s=25, alpha=0.65)
            ax.axhline(condition_rows[score].mean(), color=color, linewidth=1.5, label="Mean across session entries")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid scores", ha="center", transform=ax.transAxes)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set(title=f"{name}: one point per PCA", xlabel="Session entry (idx)", ylabel=ylabel)
    fig.tight_layout()
    return fig, axes


def plot_paired_regressions(summary, *, score="r2_adjusted"):
    """Compare tracking/playback scores matched by session idx.

    Return figure, axes, and a paired table with tracking, playback and their
    difference (tracking minus playback). Only complete finite pairs with both
    fits marked ok are used. No model fitting or significance test is performed.
    """
    import matplotlib.pyplot as plt

    if summary.index.names != ["idx", "condition"] or not summary.index.is_unique:
        raise ValueError("Use the pooled summary indexed by (idx, condition).")
    if score not in ("r2", "r2_adjusted"):
        raise ValueError("score must be 'r2' or 'r2_adjusted'.")
    valid = summary.loc[(summary["status"] == "ok") & np.isfinite(summary[score])]
    paired = valid[score].unstack("condition").reindex(columns=[0, 1]).dropna()
    paired = paired.rename(columns={0: "tracking", 1: "playback"})
    if paired.empty:
        raise ValueError("No complete finite tracking/playback pairs are available.")
    paired["difference"] = paired["tracking"] - paired["playback"]

    label = "Adjusted R-squared" if score == "r2_adjusted" else "R-squared"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for row in paired.itertuples():
        axes[0].plot([0, 1], [row.tracking, row.playback], color="gray", alpha=0.3, linewidth=0.8)
    axes[0].scatter(np.zeros(len(paired)), paired["tracking"], color="red", s=22, alpha=0.6)
    axes[0].scatter(np.ones(len(paired)), paired["playback"], color="black", s=22, alpha=0.6)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set(
        xticks=[0, 1], xticklabels=["Tracking", "Playback"], xlim=(-0.3, 1.3),
        ylabel=label, title=f"Matched sessions: {len(paired)} pairs\nOne line per session entry",
    )

    axes[1].scatter(paired.index, paired["difference"], color="#7b3294", s=25, alpha=0.7)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(
        paired["difference"].mean(), color="#7b3294", linewidth=1.5,
        label=f"Mean difference = {paired['difference'].mean():.3f}",
    )
    axes[1].set(
        xlabel="Session entry (idx)", ylabel=f"Tracking - playback ({label.lower()})",
        title="Within-session differences\nPositive = higher tracking score",
    )
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig, axes, paired


def summarize_starting_position_bias(regressions, *, midpoint=14.0):
    """Add bout-level position distribution measures to starting-position scores.

    Use observed targets in each fit's prediction table, so only the actual
    regression subset contributes. Count each bout once, regardless of windows.
    bias = (n_right - n_left) / n, with midpoint assigned to the right side.
    Standard deviation and skewness use unique bout positions, too.
    """
    if regressions["settings"]["target"] != "initial_position":
        raise ValueError("Use a regression bundle whose target is initial_position.")
    if not np.isfinite(midpoint):
        raise ValueError("midpoint must be finite.")
    summary = regressions["summary"].copy()
    fields = (
        "n_position_bouts", "n_left", "n_right", "side_bias", "abs_side_bias",
        "position_mean", "position_std", "position_skewness",
    )
    for field in fields:
        summary[field] = np.nan
    summary["position_midpoint"] = midpoint
    for key, fit in regressions["fits"].items():
        observed = fit["predictions"]["observed"]
        by_bout = observed.groupby(level="bout_id", sort=False)
        if (by_bout.nunique() > 1).any():
            raise ValueError("Starting position must be constant across windows of each bout.")
        positions = by_bout.first()
        n = len(positions)
        if not n:
            continue
        n_left = int((positions < midpoint).sum())
        n_right = int((positions >= midpoint).sum())
        bias = (n_right - n_left) / n
        std = float(positions.std(ddof=1))
        values = (
            n, n_left, n_right, bias, abs(bias), float(positions.mean()), std,
            float(positions.skew()) if n >= 3 and std > 0 else np.nan,
        )
        summary.loc[key, list(fields)] = values
    return summary


def plot_position_bias_scores(summary, *, score="r2_adjusted"):
    """Compare scores against side imbalance and position spread per condition."""
    import matplotlib.pyplot as plt

    if score not in ("r2", "r2_adjusted"):
        raise ValueError("score must be 'r2' or 'r2_adjusted'.")
    ylabel = "Adjusted R-squared" if score == "r2_adjusted" else "R-squared"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    metrics = (
        ("abs_side_bias", "Absolute side imbalance (0 = balanced, 1 = one-sided)"),
        ("position_std", "Starting-position standard deviation (position units)"),
    )
    for col, (condition, name, color) in enumerate(((0, "Tracking", "red"), (1, "Playback", "black"))):
        rows = summary.loc[(summary.index.get_level_values("condition") == condition) & (summary["status"] == "ok")]
        for row, (metric, xlabel) in enumerate(metrics):
            valid = rows.loc[np.isfinite(rows[score]) & np.isfinite(rows[metric])]
            ax = axes[row, col]
            ax.scatter(valid[metric], valid[score], color=color, s=25, alpha=0.65)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{name}: {len(valid)} session entries")
            if metric == "abs_side_bias":
                ax.set_xlim(-0.02, 1.02)
            else:
                upper = summary[metric].max()
                ax.set_xlim(0, max(float(upper) * 1.05, 1) if np.isfinite(upper) else 1)
    fig.tight_layout()
    return fig, axes


def plot_paired_bias_differences(summary, *, score="r2_adjusted"):
    """Compare matched tracking-minus-playback differences in bias and score."""
    import matplotlib.pyplot as plt

    if score not in ("r2", "r2_adjusted"):
        raise ValueError("score must be 'r2' or 'r2_adjusted'.")
    valid = summary.loc[
        (summary["status"] == "ok") & np.isfinite(summary[score]) & np.isfinite(summary["abs_side_bias"])
    ]
    scores = valid[score].unstack("condition").reindex(columns=[0, 1])
    bias = valid["abs_side_bias"].unstack("condition").reindex(columns=[0, 1])
    paired = pd.DataFrame({
        "tracking_score": scores[0], "playback_score": scores[1],
        "tracking_abs_bias": bias[0], "playback_abs_bias": bias[1],
    }).dropna()
    paired["score_difference"] = paired["tracking_score"] - paired["playback_score"]
    paired["imbalance_difference"] = paired["tracking_abs_bias"] - paired["playback_abs_bias"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(paired["imbalance_difference"], paired["score_difference"], color="#7b3294", alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ylabel = "Adjusted R-squared" if score == "r2_adjusted" else "R-squared"
    ax.set(
        xlabel="Absolute side imbalance: tracking - playback",
        ylabel=f"{ylabel}: tracking - playback", title=f"Matched session differences ({len(paired)} pairs)",
    )
    fig.tight_layout()
    return fig, ax, paired
