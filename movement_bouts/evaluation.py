"""Shared data/label helpers for representation evaluation.

See ANALYSIS_DATA_FORMAT.md. Metric functions belong here, separately from
extraction, model fitting and interactive plotting in utils.py.
"""

import numpy as np
import pandas as pd


NEURON_TYPE_COLORS = {
    "Sensorimotor": "#9BD77B",  # light green
    "Motor": "#2878B5",         # blue
    "Sensory": "#E6C62E",       # yellow
    "None": "#8B8B8B",          # gray
    "All neurons": "#303030",
}


def pair_neuron_group_scores(summary, *, score, reference_group="All neurons"):
    """Match each group's valid scores to a reference by session AND condition.

    Summary uses unique (group, idx, condition) rows. Failed/nonfinite fits are
    omitted. When supplied, observation and bout counts must match within pairs.
    Return group score, reference score, difference, and neuron/sample counts.
    """
    if summary.index.names != ["group", "idx", "condition"] or not summary.index.is_unique:
        raise ValueError("Use a unique summary indexed by (group, idx, condition).")
    if reference_group not in summary.index.get_level_values("group"):
        raise ValueError(f"Missing reference group: {reference_group}")
    valid = summary.loc[summary["status"].eq("ok") & np.isfinite(summary[score])]
    fields = [score] + [c for c in ("n_neurons", "n_observations", "n_bouts") if c in summary]
    reference = valid.loc[valid.index.get_level_values("group") == reference_group, fields].droplevel("group")
    reference = reference.rename(columns={c: "reference_" + c for c in fields})
    pairs = []
    for group in summary.index.get_level_values("group").unique():
        if group == reference_group:
            continue
        selected = valid.loc[valid.index.get_level_values("group") == group, fields].droplevel("group")
        matched = selected.join(reference, how="inner", validate="one_to_one")
        for count in ("n_observations", "n_bouts"):
            if count in matched and not matched[count].eq(matched["reference_" + count]).all():
                raise ValueError(f"Matched {group}/{reference_group} fits have different {count}.")
        matched = matched.rename(columns={score: "group_score", "reference_" + score: "reference_score"})
        matched["difference"] = matched["group_score"] - matched["reference_score"]
        matched["group"] = group
        pairs.append(matched.reset_index().set_index(["group", "idx", "condition"]))
    if pairs:
        return pd.concat(pairs).sort_index()
    return pd.DataFrame(columns=["group_score", "reference_score", "difference"],
                        index=pd.MultiIndex.from_arrays([[], [], []], names=["group", "idx", "condition"]))


def plot_neuron_group_comparison(summary, *, score, reference_group="All neurons", group_colors=None):
    """Plot matched scores/differences, with neuron-type colors in both conditions."""
    import matplotlib.pyplot as plt

    pairs = pair_neuron_group_scores(summary, score=score, reference_group=reference_group)
    groups = [g for g in summary.index.get_level_values("group").unique() if g != reference_group]
    colors = {**NEURON_TYPE_COLORS, **(group_colors or {})}
    label = {"r2": "R-squared", "r2_adjusted": "Adjusted R-squared",
             "balanced_accuracy": "Balanced accuracy"}.get(score, score)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for col, (condition, name) in enumerate(((0, "Tracking"), (1, "Playback"))):
        rows = pairs.loc[pairs.index.get_level_values("condition") == condition]
        markers = ("o", "s", "^", "D", "v", "P")
        for i, group in enumerate(groups):
            color = colors.get(group, "#666666")
            selected = rows.loc[rows.index.get_level_values("group") == group]
            axes[0, col].scatter(selected["reference_score"], selected["group_score"],
                                 color=color, edgecolors="0.25", linewidths=.4,
                                 marker=markers[i % len(markers)], s=30, alpha=.85,
                                 label=f"{group} (n={len(selected)})")
            # Fixed offsets only separate coincident points; they encode no data.
            offsets = np.linspace(-.12, .12, len(selected)) if len(selected) > 1 else np.zeros(len(selected))
            axes[1, col].scatter(i + offsets, selected["difference"], color=color,
                                edgecolors="0.25", linewidths=.4, s=24, alpha=.85)
            if len(selected):
                axes[1, col].plot([i-.22, i+.22], [selected["difference"].mean()]*2, color=color, linewidth=2)
        if len(rows):
            lo = min(rows["reference_score"].min(), rows["group_score"].min())
            hi = max(rows["reference_score"].max(), rows["group_score"].max())
            pad = max((hi-lo)*.07, .005)
            axes[0, col].plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="0.5", linewidth=.8)
        else:
            axes[0, col].text(.5, .5, "No valid matched scores", ha="center", transform=axes[0, col].transAxes)
        axes[0, col].set(title=f"{name}: matched session scores", xlabel=f"{reference_group}: {label}",
                          ylabel=f"Neuron type: {label}")
        axes[0, col].legend(fontsize=8)
        axes[1, col].axhline(0, color="0.5", linestyle="--", linewidth=.8)
        counts = [int((rows.index.get_level_values("group") == group).sum()) for group in groups]
        axes[1, col].set(xticks=range(len(groups)),
                          xticklabels=[f"{group}\nn={n}" for group, n in zip(groups, counts)],
                          ylabel=f"Type - {reference_group} ({label})",
                          title="One dot per matched session; bars = paired mean difference")
        for tick, group in zip(axes[1, col].get_xticklabels(), groups):
            tick.set_bbox(dict(facecolor=colors.get(group, "#666666"), edgecolor="none", alpha=.4, pad=2))
    return fig, axes, pairs


def plot_neuron_group_pairwise(summary, *, score, group_colors=None):
    """Heatmaps of paired mean score differences, with explicit pair counts.

    Each cell matches that pair's own eligible valid sessions within condition.
    Positive values favor the row group. No common-all-groups restriction is used.
    """
    import matplotlib.pyplot as plt

    groups = summary.index.get_level_values("group").unique().tolist()
    colors = {**NEURON_TYPE_COLORS, **(group_colors or {})}
    records = []
    for reference in groups:
        pairs = pair_neuron_group_scores(summary, score=score, reference_group=reference)
        for condition in (0, 1):
            for group in groups:
                if group == reference:
                    rows = summary.loc[(summary.index.get_level_values("group") == group)
                                       & (summary.index.get_level_values("condition") == condition)]
                    n = int((rows["status"].eq("ok") & np.isfinite(rows[score])).sum())
                    mean = 0.0 if n else np.nan
                else:
                    rows = pairs.loc[(pairs.index.get_level_values("group") == group)
                                     & (pairs.index.get_level_values("condition") == condition)]
                    n = len(rows)
                    mean = rows["difference"].mean() if n else np.nan
                records.append(dict(condition=condition, group_a=group, group_b=reference,
                                    n_pairs=n, mean_difference=mean))
    table = pd.DataFrame(records).set_index(["condition", "group_a", "group_b"])
    values = table["mean_difference"].dropna().abs()
    limit = max(float(values.max()) if len(values) else 0, .001)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, condition, name in zip(axes, (0, 1), ("Tracking", "Playback")):
        rows = table.xs(condition, level="condition")
        matrix = rows["mean_difference"].unstack("group_b").reindex(index=groups, columns=groups)
        counts = rows["n_pairs"].unstack("group_b").reindex(index=groups, columns=groups)
        plotted = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit)
        for i in range(len(groups)):
            for j in range(len(groups)):
                value, n = matrix.iloc[i, j], int(counts.iloc[i, j])
                text = f"{value:+.3f}\nn={n}" if np.isfinite(value) else "No pairs\nn=0"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(value) and abs(value) > limit*.6 else "black")
        ax.set(xticks=range(len(groups)), xticklabels=groups, yticks=range(len(groups)),
               yticklabels=groups, title=name, xlabel="Column group", ylabel="Row group")
        ax.tick_params(axis="x", labelrotation=35)
        # Group identity belongs to the labels; cell colors still encode differences.
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, group in zip(ticks, groups):
                tick.set_bbox(dict(facecolor=colors.get(group, "#666666"), edgecolor="none", alpha=.4, pad=2))
    fig.colorbar(plotted, ax=axes, label=f"Paired mean difference in {score}: row - column", shrink=.8)
    return fig, axes, table


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
            ("representation_settings" if pca_cache.get("representation") == "UMAP" else "pca_settings"):
                dict(pca_cache.get("settings", {})),
            "representation": pca_cache.get("representation", "PCA"),
        },
    }


def _annotate_session_animals(ax, summary):
    """Label animal/headstage blocks from actual session metadata, including skips."""
    if "animal" not in summary.columns:
        return
    fields = ["idx", "animal"] + (["headstage"] if "headstage" in summary else [])
    sessions = summary.reset_index()[fields].drop_duplicates().sort_values("idx")
    if sessions.empty:
        return
    if sessions["idx"].duplicated().any():
        raise ValueError("Each session idx must identify exactly one animal/headstage.")
    positions = sessions["idx"].to_numpy(dtype=float)
    animals = sessions["animal"].fillna("Unknown").astype(str).to_numpy()
    starts = np.r_[0, np.flatnonzero(animals[1:] != animals[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(animals) - 1]
    centers, labels = [], []
    for group, (start, end) in enumerate(zip(starts, ends)):
        left = (positions[start-1] + positions[start])/2 if start else positions[start] - .5
        right = (positions[end] + positions[end+1])/2 if end+1 < len(animals) else positions[end] + .5
        if group % 2 == 0:
            ax.axvspan(left, right, color=".5", alpha=.08, zorder=0)
        if start:
            ax.axvline(left, color=".55", linestyle=":", linewidth=1, zorder=0)
        centers.append((left + right)/2)
        labels.append(animals[start])
    has_headstages = "headstage" in sessions
    animal_axis = ax.secondary_xaxis(1.065 if has_headstages else 1.0)
    animal_axis.set_xticks(centers, labels)
    animal_axis.tick_params(axis="x", length=0, labelsize=9, pad=4)
    animal_axis.spines["top"].set_visible(False)
    if has_headstages:
        headstages = sessions["headstage"].fillna("?").astype(str).to_numpy()
        changes = (animals[1:] != animals[:-1]) | (headstages[1:] != headstages[:-1])
        hs_starts = np.r_[0, np.flatnonzero(changes) + 1]
        hs_ends = np.r_[hs_starts[1:] - 1, len(animals) - 1]
        hs_centers, hs_labels = [], []
        for start, end in zip(hs_starts, hs_ends):
            left = (positions[start-1] + positions[start])/2 if start else positions[start] - .5
            right = (positions[end] + positions[end+1])/2 if end+1 < len(animals) else positions[end] + .5
            if start and animals[start] == animals[start-1]:
                ax.axvline(left, color=".65", linestyle="--", linewidth=.8, zorder=0)
            hs_centers.append((left + right)/2)
            hs_labels.append(f"hs{headstages[start]}")
        hs_axis = ax.secondary_xaxis("top")
        hs_axis.set_xticks(hs_centers, hs_labels)
        hs_axis.tick_params(axis="x", length=0, labelsize=8, labelcolor=".4", pad=3)
        hs_axis.spines["top"].set_visible(False)
    ax.set_title(ax.get_title(), pad=46 if has_headstages else 28)


def plot_regression_summary(summary, *, score="r2_adjusted", representation="PCA"):
    """Plot linear/logistic scores per session/condition, without window splitting."""
    import matplotlib.pyplot as plt

    if summary.index.names != ["idx", "condition"] or not summary.index.is_unique:
        raise ValueError("Use the pooled summary indexed by (idx, condition).")
    rows = summary.reset_index()
    rows = rows.loc[(rows["status"] == "ok") & np.isfinite(rows[score])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    ylabel = {
        "r2": "Descriptive R-squared", "r2_adjusted": "Adjusted R-squared",
        "mae": "MAE", "balanced_accuracy": "Balanced accuracy",
    }.get(score, score)
    reference = 0.5 if score == "balanced_accuracy" else 0
    for ax, (condition, name, color) in zip(axes, ((0, "Tracking", "red"), (1, "Playback", "black"))):
        condition_rows = rows.loc[rows["condition"] == condition].sort_values("idx")
        if len(condition_rows):
            ax.scatter(condition_rows["idx"], condition_rows[score], color=color, s=25, alpha=0.65)
            ax.axhline(condition_rows[score].mean(), color=color, linewidth=1.5, label="Mean across session entries")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid scores", ha="center", transform=ax.transAxes)
        ax.axhline(reference, color="gray", linestyle="--", linewidth=0.8)
        ax.set(title=f"{name}: one point per {representation}", xlabel="Session entry (idx)", ylabel=ylabel)
        if score == "balanced_accuracy":
            ax.set_ylim(-0.02, 1.02)
        _annotate_session_animals(ax, summary)
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
    if score not in ("r2", "r2_adjusted", "balanced_accuracy"):
        raise ValueError("score must be 'r2', 'r2_adjusted' or 'balanced_accuracy'.")
    valid = summary.loc[(summary["status"] == "ok") & np.isfinite(summary[score])]
    paired = valid[score].unstack("condition").reindex(columns=[0, 1]).dropna()
    paired = paired.rename(columns={0: "tracking", 1: "playback"})
    if paired.empty:
        raise ValueError("No complete finite tracking/playback pairs are available.")
    paired["difference"] = paired["tracking"] - paired["playback"]

    label = {
        "r2": "R-squared", "r2_adjusted": "Adjusted R-squared",
        "balanced_accuracy": "Balanced accuracy",
    }[score]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for row in paired.itertuples():
        axes[0].plot([0, 1], [row.tracking, row.playback], color="gray", alpha=0.3, linewidth=0.8)
    axes[0].scatter(np.zeros(len(paired)), paired["tracking"], color="red", s=22, alpha=0.6)
    axes[0].scatter(np.ones(len(paired)), paired["playback"], color="black", s=22, alpha=0.6)
    axes[0].axhline(0.5 if score == "balanced_accuracy" else 0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set(
        xticks=[0, 1], xticklabels=["Tracking", "Playback"], xlim=(-0.3, 1.3),
        ylabel=label, title=f"Matched sessions: {len(paired)} pairs\nOne line per session entry",
    )
    if score == "balanced_accuracy":
        axes[0].set_ylim(-0.02, 1.02)

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
    _annotate_session_animals(axes[1], summary)
    fig.tight_layout()
    return fig, axes, paired


def fit_direction_logistic(data, labels, *, C=1.0):
    """Fit a descriptive left/right classifier to aligned bout/window tables.

    Keep initial_direction -1 (left) and +1 (right); exclude all other values.
    Fit StandardScaler and L2 logistic regression (intercept, balanced class
    weights) together in a saved Pipeline. C controls inverse penalty strength.
    Return balanced accuracy, both direction recalls, and indexed predictions
    with observed/predicted direction and probability_right. Scores use the
    fitted observations; no PCA fitting, holdout, or randomization is performed.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    validate_data_labels(data, labels)
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a finite positive number.")
    keep = labels["initial_direction"].isin([-1, 1])
    x = data.loc[keep]
    y = labels.loc[keep, "initial_direction"].astype(int)
    if y.nunique() != 2:
        raise ValueError("Both left (-1) and right (+1) observations are required.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-6),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, y)
        except ConvergenceWarning as error:
            raise ValueError("Logistic regression did not converge.") from error
    predicted = model.predict(x)
    right_column = list(model.classes_).index(1)
    predictions = pd.DataFrame({
        "observed": y, "predicted": predicted,
        "probability_right": model.predict_proba(x)[:, right_column],
    }, index=x.index)
    return {
        "model": model, "target": "initial_direction", "C": C,
        "n_features": x.shape[1], "n_excluded": int((~keep).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "recall_left": float((predicted[y == -1] == -1).mean()),
        "recall_right": float((predicted[y == 1] == 1).mean()),
        "predictions": predictions,
    }


def run_direction_logistics(pca_cache, *, columns, C=1.0):
    """Fit one left/right logistic regression per session/condition, all windows.

    Use saved coordinates and only known directions, without displacement or
    position filtering. Return summary/fits/settings; keys are (idx, condition).
    Keep failed subsets as skipped rows with a reason; do not stop other fits.
    Counts distinguish bout/window observations from unique bouts. No file I/O.
    """
    if isinstance(columns, str):
        raise TypeError("columns must be a sequence of feature names.")
    columns = list(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Supply a nonempty sequence of unique feature names.")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a finite positive number.")
    records, fits = [], {}
    for idx, condition_fits in pca_cache["sessions"].items():
        info = pca_cache["session_info"].loc[idx]
        for condition, pca_fit in condition_fits.items():
            rows = pca_fit["results"]
            record = {
                "idx": idx, "condition": condition,
                "animal": info["animal"], "headstage": info["headstage"],
                "session": info["session"], "region": info.get("region", None),
                "target": "initial_direction", "n_features": len(columns),
                "time_start_s": float(rows.index.get_level_values("bin_start_s").min()),
                "time_end_s": float(rows["bin_end_s"].max()),
                "n_windows": int(rows["bin_number"].nunique()),
                "n_candidates": len(rows), "n_excluded_target": 0,
                "n_observations": 0, "n_bouts": 0,
                "n_left_observations": 0, "n_right_observations": 0,
                "n_left_bouts": 0, "n_right_bouts": 0,
                "balanced_accuracy": np.nan, "recall_left": np.nan, "recall_right": np.nan,
                "status": "skipped", "reason": "",
            }
            missing = (set(columns) | {"initial_direction"}).difference(rows.columns)
            if missing:
                record["reason"] = "Missing columns: " + ", ".join(sorted(missing))
                records.append(record)
                continue
            valid = rows["initial_direction"].isin([-1, 1])
            selected = rows.loc[valid]
            record["n_excluded_target"] = int((~valid).sum())
            record["n_observations"] = len(selected)
            record["n_bouts"] = selected.index.get_level_values("bout_id").nunique()
            for direction, name in ((-1, "left"), (1, "right")):
                direction_rows = selected.loc[selected["initial_direction"] == direction]
                record[f"n_{name}_observations"] = len(direction_rows)
                record[f"n_{name}_bouts"] = direction_rows.index.get_level_values("bout_id").nunique()
            try:
                fit = fit_direction_logistic(rows[columns], rows, C=C)
            except ValueError as error:
                record["reason"] = str(error)
            else:
                fits[(idx, condition)] = fit
                record.update({name: fit[name] for name in ("balanced_accuracy", "recall_left", "recall_right")})
                record["status"] = "ok"
            records.append(record)
    if not records:
        raise ValueError("No session/condition subsets are available to evaluate.")
    return {
        "summary": pd.DataFrame(records).set_index(["idx", "condition"]),
        "fits": fits,
        "settings": {
            "target": "initial_direction", "columns": columns, "C": C,
            "penalty": "l2", "standardized": True, "class_weight": "balanced",
            "solver": "lbfgs", "max_iter": 2000, "tol": 1e-6,
            "valid_directions": [-1, 1], "window_mode": "pooled",
            "min_displacement": None, "scoring": "descriptive_in_sample",
            "projection_basis": pca_cache.get("projection_basis", "separate_condition_pcas"),
            ("representation_settings" if pca_cache.get("representation") == "UMAP" else "pca_settings"):
                dict(pca_cache.get("settings", {})),
            "representation": pca_cache.get("representation", "PCA"),
        },
    }


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
