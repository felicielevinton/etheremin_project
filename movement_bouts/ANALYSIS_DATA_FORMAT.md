# Shared format for neuronal data, PCA, UMAP and evaluation

Analysis and evaluation use **two aligned pandas DataFrames**, `data` and
`labels`. The current notebook calls the neuronal input pair `pca_input` and
`pca_labels`; the names can change without changing the format.

## Observations and their identity

Each row is **one movement bout at one premotor time window**. Both tables use
the same unique MultiIndex, with these names and this order:

```text
bout_id   bin_start_s
12        -0.5
12        -0.4
12        -0.3
12        -0.2
12        -0.1
18        -0.5
...
```

- `bout_id` is the original `movement_bouts` row index. Never reset or renumber
  it after session selection, filtering or concatenation. It refers to a specific
  saved bout-table export, not an identifier stable across regenerated exports.
- `bin_start_s` is relative to movement onset. Windows are start-inclusive and
  end-exclusive. Five 100 ms windows span `[-0.5, 0)` seconds; onset is excluded.
- Rows normally follow each bout's windows chronologically. If reordered, reorder
  both tables identically. `data.index.equals(labels.index)` must be true.
- A bout's windows share behavioral labels and recording context; they are not
  independent bouts. Permutations and resampling must respect bout identity.

## The two tables

| Table | Rows | Columns |
| --- | --- | --- |
| `data` | Bout/window observations | Only numeric features or coordinates |
| `labels` | Exactly the same index and row order | Behavioral and observation metadata |

Examples of `data` columns:

- Original neuronal responses: neuron IDs, values in spikes/s after temporal averaging.
- PCA: `PC1`, `PC2`, ... (all retained coordinates are saved).
- A future UMAP analysis can use `UMAP1`, `UMAP2`, ... with the same row index.

Neural arrays are stacked as `(bouts, neurons, windows)`, then transposed to
`(bouts, windows, neurons)` before forming the observation table. For the inspected
session 0, 347 bouts, 14 neurons and five windows give `data.shape == (1735, 14)`.

`labels` currently retains all columns from `movement_bouts`, except the neural
`spike_counts` column added by the notebook join. Important fields are:

| Field | Meaning |
| --- | --- |
| `idx` | Session-entry ID linking to `session_info.loc[idx]` |
| `animal`, `headstage`, `session` | Source recording identifiers |
| `condition` | 0 = tracking, 1 = playback, read at movement onset |
| `initial_position` | Position at movement onset |
| `initial_direction` | -1 = left, +1 = right; 0/unknown is not either direction |
| `final_position` | Observed movement endpoint position; missing for unfinished bouts |
| `total_movement`, `mean_speed`, onset/end fields | Original bout measurements |
| `bin_number` | Chronological window number, starting at 0 |
| `bin_end_s` | Exclusive window end relative to movement onset |

Bout measurements are repeated across windows, not recalculated within each
premotor window. In particular, `mean_speed` describes the detected movement
bout, not its preceding premotor window. Position units need verification before
calling them cm. `final_position` is an observed endpoint, not a planned target.

## Current saved PCA results

The file `data/derived/premotor_activity/pca/pca_5bins.pkl` (workspace-relative)
contains the model/result bundle. `N_BINS` determines the filename suffix.

```python
cache = pd.read_pickle(premotor_dir / "pca" / f"pca_{N_BINS}bins.pkl")
fit = cache["sessions"][idx][condition]  # condition: 0 or 1
model = fit["model"]
results = fit["results"]
```

`results` combines labels and PC columns for convenient plotting. Select the
columns needed for an evaluation directly:

```python
data = results[["PC1", "PC2", "PC3"]]
labels = results[[
    "idx", "condition", "initial_position", "final_position",
    "initial_direction", "bin_number", "bin_end_s",
]]
```

There is no required conversion or extra saved dataset. Before fitting PCA, the
notebook already has separate `pca_input` and `pca_labels` tables. After fitting,
select explicit coordinate and target columns from `results`. Other representations
can follow the same index convention with their own coordinate names.

Other bundle fields include `settings`, `session_info`, and fixed `position_limits`.
Settings record binning, preprocessing and source-extraction information. Models
retain neuron order, mean responses, loadings and explained variance.

The tracking-projection bundle (`pca_tracking_projection_5bins.pkl`) uses the same
structure and adds `projection_basis="tracking"`: both conditions use the saved
tracking model's centering and loadings. Ordinary bundles fit each condition
separately. Coordinates from separately fitted sessions/conditions have different
bases, so keep their evaluation results identified by session and condition.

## Filtering and evaluation conventions

```python
from evaluation import validate_data_labels

mask = (labels["bin_number"] == 4) & labels["initial_direction"].isin([-1, 1])
window_data = data.loc[mask]
window_labels = labels.loc[mask]
validate_data_labels(window_data, window_labels)
```

- Numeric data must be finite. Missing behavioral labels are allowed in the
  format; an evaluation excludes missing values only for its required targets.
- Always apply the same mask to both tables. Do not reset their indices.
- For per-window metrics, keep the MultiIndex when selecting a window.
- Describe every result by session, condition, representation/basis, dimensions,
  time window, filters and sample counts. A score belongs to that chosen space.
- Metric helpers belong in `evaluation.py` and accept the aligned pair. Keep
  extraction, embedding fitting and plotting separate. Descriptive label regression
  is implemented; silhouette separation and randomization controls remain to add.
- Save per-observation outputs with the same index. Save aggregate scores in a
  separate summary table, with their analysis settings. Existing PCA bundles do
  not need rewriting to follow this interface.

## Descriptive regression

`evaluation.fit_label_regression(data, labels, target="initial_position")` fits
ordinary linear regression with an intercept to one numeric continuous label.
Use explicit features/PCs and masks before calling; the helper does not group
sessions, conditions or windows automatically. It also works directly with the
existing `pca_input` and `pca_labels` pair.

The result dictionary contains `model`, `target`, descriptive `r2`, `mae`,
`n_features`, `df_residual`, `r2_adjusted`, `n_excluded`, and `predictions`. The latter retains the used observation index
with `observed`, `fitted`, and `residual` (observed minus fitted) columns. Only
rows with missing/nonfinite target values are omitted. A nonconstant target and
at least two valid observations are required. Other missing labels are irrelevant.

`plot_label_regression(result)` draws observed versus fitted values and an
identity line without fitting anything. Both metrics describe the same data used
to fit, with no standardization, cross-validation or randomization. They do not
establish significance or prediction of new bouts. Compare the same feature
counts/windows when interpreting scores; adding predictors can improve an
in-sample fit without indicating a stronger biological relationship.

## Batch regression and saved evaluation results

`run_label_regressions(cache, target=..., columns=[...], min_displacement=None)`
fits **one pooled regression per session and condition**, using every premotor
window together. Windows remain separate observations; they are not averaged.
An optional cutoff retains `abs(final_position - initial_position) > cutoff`,
including both displacement directions and excluding unknown endpoints.

The returned bundle has three parts:

- `summary`: a DataFrame indexed by `(idx, condition)`, one row per PCA. It records
  session identifiers, covered time interval, number of windows, target, counts,
  R2, adjusted R2, MAE, residual degrees of freedom, status and skip reason.
  `n_candidates` is before filtering; `n_filtered` counts displacement exclusions;
  `n_excluded_target` counts invalid targets after filtering. `n_observations`
  counts bout/window rows, while `n_bouts` counts distinct bouts. Skipped subsets
  retain their counts but have no fit scores or model.
- `fits`: a dictionary keyed by `(idx, condition)`. Successful entries contain
  regression models and prediction tables retaining **all** original
  `(bout_id, bin_start_s)` observation IDs.
- `settings`: target, feature columns, cutoff, `window_mode="pooled"`, source PCA
  settings and representation basis.

Residual degrees of freedom are `n_observations - rank(design including intercept)`.
Adjusted R2 is `1 - (1-R2)*(n_observations-1)/df_residual`, and is NaN for zero
residual degrees of freedom. These are row-based descriptive diagnostics: they
**do not adjust for dependence among windows from the same bout** and must not be
read as counts of independent bouts or held-out performance.

Section 18 saves the bundle with `pd.to_pickle()` in
`data/derived/premotor_activity/evaluation/` (workspace-relative). Filenames end
in `_pooled.pkl`, distinguishing them from the superseded window-wise analysis.
Reload with `pd.read_pickle(path)`; `plot_regression_summary(bundle["summary"])`
shows one dot per session/PCA, tracking and playback in separate panels. Horizontal
colored lines show the unweighted mean across session entries. No time-course,
confidence interval, resampling, randomization or PCA refitting is introduced.

`plot_paired_regressions(summary, score="r2_adjusted")` matches the two conditions
by session idx and returns `(figure, axes, paired_table)`. The table contains
`tracking`, `playback`, and `difference` (tracking minus playback). Both fits must
be successful with finite scores; incomplete pairs are excluded. Select "r2" for
ordinary R2. This plots saved results without model fitting or significance tests.
