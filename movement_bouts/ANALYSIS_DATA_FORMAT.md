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

- Neuronal responses: neuron IDs, values in spikes/s after temporal averaging,
  optionally expressed relative to each bout/neuron's earlier baseline rate.
- PCA: `PC1`, `PC2`, ... (all retained coordinates are saved).
- A future UMAP analysis can use `UMAP1`, `UMAP2`, ... with the same row index.

Neural arrays are stacked as `(bouts, neurons, windows)`, then transposed to
`(bouts, windows, neurons)` before forming the observation table. For the inspected
session 0, 347 bouts, 14 neurons and five windows give `data.shape == (1735, 14)`
in both raw and baseline-subtracted modes.

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

## Extraction, analysis windows and baseline subtraction

The notebook constants select three intervals relative to movement onset:

| Setting | Default | Meaning |
| --- | --- | --- |
| EXTRACTION_WINDOW_S | (-1.2, 0.0) | Raw counts retained at 5 ms resolution |
| ANALYSIS_WINDOW_S | (-0.5, 0.0) | Interval averaged and passed into PCA |
| BASELINE_WINDOW_S | (-1.0, -0.5) | Mean rate used for optional correction |
| ACTIVITY_MODE | "raw" | Alternatively "baseline_subtracted" |
| N_BINS | 5 | Number of equal windows within ANALYSIS_WINDOW_S |

All intervals are start-inclusive/end-exclusive. `extracted_-1.2_0/` under
`data/derived/premotor_activity/` contains neuronal_bouts.pkl, neuron_ids_by_session.pkl,
excluded_bouts.pkl, and parameters.json. Each new raw array is (neurons, 240).
Metadata records its actual `window_s` and `DT`; use those values when loading.
The earlier 500 ms extractions and results at the root remain prior outputs.
Incomplete extractions are skipped with their original bout IDs and a reason.

`prepare_pca_input(..., idx=..., dt=..., n_bins=..., extraction_window_s=...,
analysis_window_s=..., baseline_window_s=..., activity_mode=...)` first selects
the analysis interval, then averages its counts into firing rates. The caller
passes the extraction interval and DT from the saved extraction metadata.
N_BINS never divides the wider extraction merely because more counts were saved.

In baseline_subtracted mode, each bout and neuron has one baseline rate:
total baseline spikes / baseline duration. Subtract that value from each of its
analysis windows. Negative values mean activity below that baseline, in spikes/s.
This is subtraction, not division or variance standardization. Baseline means are
computed from retained counts, so changing the baseline within the saved interval
needs no original recording reload. No extra behavioral columns or baseline rows
are inserted into the observation tables. Raw mode does not require a baseline.

Windows must align to the original DT and fit within the saved extraction;
the baseline must end at/before analysis start. N_BINS must divide the number of
original bins in the analysis interval. The same retained bouts are available to
both modes. No additional immobility test or displacement filter is introduced.

## Current saved PCA results

The selected `run_dir/pca/pca_5bins.pkl` contains the model/result bundle.
The notebook constructs run_dir once under `data/derived/premotor_activity/runs/`:

- `analysis_-0.5_0_5bins_raw/`
- `analysis_-0.5_0_5bins_baseline_subtracted_-1_-0.5/`

These names identify analysis bounds, bin count, mode, and baseline bounds when
used. Each run has pca/ and evaluation/ directories; viewers and evaluations use
the central pca_path/evaluation_dir variables. Existing root-level caches remain
available as prior results. Changing preprocessing requires new PCA/evaluation
fits but no re-extraction while requested windows remain within the saved range.

```python
cache = pd.read_pickle(pca_path)
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
Settings additionally record activity_mode, analysis_window_s, baseline_window_s
(None for raw), extraction_directory and embedded extraction_parameters.

The tracking-projection bundle (`pca_tracking_projection_5bins.pkl`) uses the same
structure and adds `projection_basis="tracking"`: both conditions use the saved
tracking model's centering and loadings. Ordinary bundles fit each condition
separately. Coordinates from separately fitted sessions/conditions have different
bases, so keep their evaluation results identified by session and condition.
Projection preparation uses the reference tracking PCA's saved preprocessing and
extraction settings, including its baseline, rather than current notebook constants.

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
the selected run's `evaluation/` directory. Filenames end
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

## Descriptive left/right logistic regression

`fit_direction_logistic(data, labels, C=1.0)` uses the same aligned tables and
observation IDs. Only `initial_direction` -1 (leftward) and +1 (rightward) are
included; 0/missing/unknown values are excluded. Both directions are required.
This label describes movement direction, not the side of the starting position.

The saved `model` is a Pipeline containing StandardScaler followed by logistic
regression with an intercept, L2 regularization, inverse penalty strength C,
and `class_weight="balanced"`. Scaling is fitted to the selected coordinates
within each classifier; saved PCA models and coordinates are unchanged.
See the [scikit-learn logistic regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Results include `balanced_accuracy`, `recall_left`, `recall_right`, `n_excluded`,
and an indexed `predictions` table with `observed`, `predicted`, and
`probability_right`. Balanced accuracy is the mean of the two direction recalls;
a constant classifier scores 0.5 when both directions are present. Scores are
computed on the fitted observations, without held-out evaluation or randomization.

`run_direction_logistics(cache, columns=[...], C=1.0)` fits one classifier per
session/condition, pooling all windows and applying no displacement filter.
It returns `summary`, `fits`, and `settings` like the linear regression batch.
The summary records observation and unique-bout counts, including left/right
counts separately, scores, status and skip reason. Missing columns, one-class
subsets, and convergence failures are recorded as skipped. Models and prediction
tables are keyed by `(idx, condition)`; settings retain scaling, penalty, weighting,
solver, source PCA settings, and representation basis.

Notebook section 22 saves this bundle as `evaluation/logistic_initial_direction_...
_pooled.pkl` under the premotor activity directory. Existing plotting helpers
`plot_regression_summary` and `plot_paired_regressions` accept
`score="balanced_accuracy"`. Absolute-score plots have a 0.5 reference; the
tracking-minus-playback difference plot has a zero reference. No model fitting
occurs in either plotting helper.
