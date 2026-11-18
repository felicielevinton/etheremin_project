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

    Return the fitted model, target name, in-sample R2/MAE, n_excluded, and a
    predictions table (observed, fitted, residual = observed - fitted).
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
    return {
        "model": model,
        "target": target,
        "r2": float(r2_score(y, fitted)),
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
