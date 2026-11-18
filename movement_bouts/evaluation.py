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
