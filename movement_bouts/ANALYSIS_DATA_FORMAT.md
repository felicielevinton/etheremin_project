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
| `early_mean_speed` | Mean absolute speed over the capped early interval |
| `peak_speed` | Maximum absolute speed over that same early interval |
| `peak_speed_latency_s` | Time after onset of the earliest maximum |
| `early_speed_duration_s` | Actual observed length of the early interval |
| `early_speed_complete` | The capped interval was recorded in full, including an earlier known bout end |
| `bin_number` | Chronological window number, starting at 0 |
| `bin_end_s` | Exclusive window end relative to movement onset |

Bout measurements are repeated across windows, not recalculated within each
premotor window. In particular, `mean_speed` describes the detected movement
bout, not its preceding premotor window. Position units need verification before
calling them cm. `final_position` is an observed endpoint, not a planned target.

The extraction notebook's selectable `EARLY_SPEED_WINDOW_S` defaults to 1.0 s.
Both early speed summaries use `[onset, min(onset + EARLY_SPEED_WINDOW_S, end))`,
where end is the next qualifying immobility start under the existing bout
definition. Samples come from `abs(Speed_x * SPEED_X_FACTOR)` in position units/s;
no new smoothing is applied. Peak latency uses the earliest maximum on ties and
is strictly before the interval stop. `mean_speed` remains the whole-bout mean.
For unfinished bouts, early values can still be measured when the full cap is
recorded. If the recording ends too soon, the observed duration is saved and
`early_speed_complete=False`; early mean/peak/latency remain missing. No bouts are
discarded by this computation. The chosen initial evaluation targets are
`initial_position`, `early_mean_speed` and `peak_speed`; other labels are retained.

`save_movement_bout_tables` archives a replaced bout pickle by its SHA-256 under
`movement_bouts/source_versions/` and keeps unchanged table files byte-identical.
For existing extraction/model fingerprints, `validate_bout_sources` permits a
column-only addition after verifying the archived hash and exact original values,
dtypes and row/index order. Other source changes require regeneration. Historical
models keep their original provenance and do not automatically gain new labels;
new evaluations should select current labels by their original bout IDs.

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
The superseded root-level 500 ms extraction tables and metadata were removed
after verifying they matched the final 100 bins of the wider extraction.
Root-level PCA, evaluation, and figure outputs remain as prior results.
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

## Neuron-type subset runs

`neuron_type_pca.ipynb` keeps this same observation/model/evaluation format.
After `prepare_pca_input`, it selects neuron columns by exact classification IDs
before fitting. Its default Sensorimotor selection requires at least five matching
neurons per session; the type, cutoff, activity mode, analysis and baseline windows
are editable in one settings cell. Missing classifications are not cluster 3 (None).

Run directories use a `neuron_type_{type}_min{cutoff}_analysis_...` prefix under
`premotor_activity/runs/`. Additional PCA bundle fields are `selected_neuron_ids`,
`session_selection` (including ineligible sessions), and `fit_summary`.
Settings also retain classification-file SHA-256, type mapping, selected type and
minimum count. The existing `session_info`, `sessions`, `settings`,
`skipped_sessions` and `position_limits` fields support the same three viewers.
Position regressions and direction logistic regression use the same saved PCAs,
with all windows pooled. No displacement filter is enabled by default; the final
position regression has an optional strict absolute-displacement cutoff.

### Matched comparisons between neuron populations

`neuron_group_comparison.ipynb` independently loads the source tables and fits
each type and the full original neuron set using identical preprocessing and
observation rows. Its own settings control type groups, cutoff and activity mode;
no single-type PCA run is required. `neuron_type_pca.ipynb` links to this notebook.
All neurons includes units without a type label. Type eligibility uses the chosen
minimum neuron count; full-session references need the requested PC dimensions.
Each population retains the standard PCA/evaluation bundle structure.

`comparison_scores[target]` concatenates evaluation summaries with unique
`(group, idx, condition)` indices and an added `n_neurons` column. Skipped and
ineligible subsets remain explicit. For each target, matched successful
type/full-session predictions must have identical observation indices.
`pair_neuron_group_scores(summary, score=..., reference_group=...)` matches by
session and condition, excludes failed/nonfinite fits, and checks equal sample
counts. It returns group/reference scores, differences and neuron/sample counts
without fitting or testing significance. Negative differences favor the reference.

Every pair uses its own shared sessions. At the five-neuron cutoff, the four type
eligibilities have no common session, so an all-groups intersection is unsuitable.
`plot_neuron_group_comparison` shows matched scores and differences from the full
session; `plot_neuron_group_pairwise` shows mean row-minus-column differences and
pair counts. Neuron counts differ and are not controlled by subsampling.
Common score/difference plots use Sensorimotor light green, Motor blue, Sensory
yellow and None gray in both condition panels. Heatmap group labels use the same
palette, while cell color remains the numerical difference scale. GROUP_COLORS
is editable and saved with the comparison settings; recoloring needs no refit.

`runs/neuron_group_comparison_min{cutoff}_analysis_.../pca_groups.pkl` holds each
population's models and labeled coordinates. Its evaluation pickle contains
`settings`, `evaluations`, `scores`, `eligibility`, `paired_to_all` and `pairwise`.
Filenames identify PC columns, destination filter and logistic C; preprocessing,
classification/extraction provenance and exact neuron IDs are retained.

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

For labels added after PCA fitting, pass `bout_labels=movement_bouts`. This
optional reference table has one row per original bout ID and must include the
target, `idx` and `condition`. The helper matches IDs, checks session/condition
membership, and repeats the current target across each bout's saved windows.
It overrides stale cached values without changing the input tables or PCA fits.
Unknown bout IDs or conflicting membership fail explicitly; missing target
values are still excluded separately for each regression. The settings identify
`target_source` as `current_bout_labels` or `saved_representation_labels`.

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

Sections 23-25 of `analyse_neuronal_activity.ipynb` apply this to
`early_mean_speed` and `peak_speed`. Run section 1 followed by those sections to
reuse the chosen saved PCA, with selectable `SPEED_REGRESSION_COLUMNS` (PC1-PC3)
and `SPEED_REGRESSION_SCORE` (adjusted R2 by default). No displacement filter is
used. Both target definitions come from the saved bout parameters, including
their early window cap. Bundles retain current bout/parameter fingerprints and
the exact source PCA hash; filename digests distinguish label sources and target
definitions. Separate summary/paired PNGs and paired-score pickles save alongside
the bundles. Cached PCA files are never rewritten. These fits do not control for
starting position and remain descriptive/in-sample.

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


## Focused raw/baseline UMAP comparison

`premotor_umap.ipynb` prepares the same neural-rate/label pairs with
`prepare_pca_input`, using five windows over [-0.5, 0) s and a [-1, -0.7) s
baseline. All recorded neuron columns retain their original order. Separate
3D UMAPs are fitted per condition and activity mode, without z-scoring,
PCA preprocessing or behavioral targets. Results use `UMAP1`, `UMAP2` and `UMAP3`
joined to the unchanged labels and `(bout_id, bin_start_s)` index.

The bundle has `fits[activity_mode][condition]` entries containing `model`,
`results` and `neuron_ids`. `fit_summary` is indexed by activity_mode/condition
with observation, bout and neuron counts, neighborhood size, status and reason.
A condition with n_observations <= n_neighbors is skipped explicitly.
`settings` records session, both modes, preprocessing, UMAP parameters/version,
seed and extraction provenance; `session_info` and `position_limits` support
plotting. Files live in `runs/umap_raw_vs_baseline/`, with session, neighbors,
seed and an effective-settings digest in the filename. The notebook saves the
pickle, four-panel 3D PNG and rotatable Plotly HTML together, with a `umap3d_` filename prefix. Two independent figures reuse the same coordinates: position colors and direction colors (left blue, right red, unknown gray), both with circles. PNG/HTML filenames end in `_position` or `_direction`. Plotting only consumes saved coordinates.
Panel coordinate systems are independent; these are exploratory embeddings,
not cross-condition distances or independent observations across bout windows.


### Overall UMAP evaluations

The overall section of `premotor_umap.ipynb` fits each saved session's two
conditions separately for both raw and baseline-subtracted activity. Default
scope is all sessions; `OVERALL_SESSION_INDICES` can select a subset. Each mode
uses the existing `sessions[idx][condition]` evaluation interface with labeled
UMAP1-3 coordinates. Metadata sets `representation="UMAP"`,
`projection_basis="separate_condition_umaps"`, and `representation_settings`
(the PCA interface continues to use `pca_settings`).

`runs/umap_overall_k.../sessions/session_{idx}.pkl` checkpoints both modes'
models, neuron IDs, coordinates and fit summaries, with exact run settings.
`embedding_fit_summary.pkl` retains every mode/session/condition, including
missing or insufficient extractions. Setting random_state=None gives each
batch a unique run directory; saved models identify the realized embedding.

`evaluation/` saves separate pooled initial/final-position regression and
direction-logistic bundles per mode, with a digest of evaluation settings,
summary and paired-score PNGs, and matched tracking/playback tables. Missing
positions and unknown directions are excluded per target. Position regressions
are ordinary linear fits; direction uses the same scaler + balanced L2 logistic
pipeline as PCA. All premotor windows are pooled. These are in-sample descriptive
scores, not significance tests; adjusted R2 does not account for dependence of
windows within bouts. No displacement filter applies to initial position or
direction; final-position filtering is optional (default None).


Session-score overview plots and paired-difference session axes derive contiguous
animal and headstage blocks from the summary's `animal` and `headstage` columns.
Animal names and hs labels occupy separate rows above the numeric session axis.
Dotted lines mark animal boundaries; lighter dashed lines mark headstage changes
within an animal. Missing/failed scores do not remove source metadata boundaries.
This shared display is used by the all-neuron PCA, neuron-type PCA and UMAP
notebooks. Movement-bout heatmaps show equivalent separators in displayed-row
coordinates, preserving their existing preference for one headstage per recording.


## Session-level cell composition and UMAP score associations

`umap_cell_composition.ipynb` joins the composition audit to saved UMAP
initial-position regression scores on original `idx`. The `session_scores`
table is indexed by (activity_mode, idx, condition), with membership and source
checks, counts, classified-unit fractions, classification coverage, source
scores and behavioral sampling controls. Missing classifications stay unknown;
zero-classified sessions cannot enter the type-fraction models.

This analysis uses session-level observations rather than bout/window rows.
`fit_session_composition_associations` therefore requires a unique session index
and runs separately per activity-mode/condition. Models compare one type fraction
at a time to a controls-only model on identical rows. Output retains Pearson and
Spearman correlations, coefficients per 10 percentage points, partial correlation,
controls/full across-session R2 and their difference, rank/df/skip reasons,
models and indexed fitted/residual values. No p-values or confidence intervals
are introduced. These association-model R2 values are distinct from the source
UMAP starting-position regression R2 used as the outcome.

Paired tables use one tracking-minus-playback score per session and differences
in behavioral sampling controls. A fully classified-session sensitivity check is
saved separately. All outputs live in data/derived/umap_composition/{digest}/,
with analysis.pkl, summary tables, provenance JSON and PNGs. Source fits remain
unchanged. Results describe associations for the chosen saved UMAP realization,
not causal effects or significance across independent animals.


## Onset-aligned PC time courses

`principal_component_psth.ipynb` and `pc_psth.py` reuse each activity mode's
saved tracking PCA. Both conditions use its mean and loadings; exact original
recording neuron IDs/order are required. Raw/corrected modes use their own
saved bases, whose component identities and signs need not agree. Do not pool
PC labels across sessions.

`project_pc_psths` returns labels/excluded_bouts with original bout indices,
centers_s, neuron_ids, settings, and modes[mode][conditions][condition] entries:
bout_ids, bout_scores (bouts x time x components), n_bouts, mean and sem
(time x components). Each mode also retains its tracking model. Baseline
subtraction acts on raw neural rates per bout/unit before smoothing and the
fixed PCA transform; no PC-level baseline subtraction is added. The saved
PCA training mean can give corrected scores a nonzero baseline.

Default extraction [-1.2, 1) includes baseline [-1, -.7) and smoothing margins;
display [-.5, .5), native 5 ms bins, Gaussian sigma .01 s. SEM is across bouts;
it is not a confidence interval across independent animals. Displayed post-onset
activity is outside the PCA training interval. In-memory results retain PCA paths/hashes, preprocessing settings, session
identity and direction selection. Figures display in the notebook, with no
automatic file saving. Original fits remain unchanged.


### In-memory PC PSTH precomputation

The PC PSTH notebook builds `pc_psth_cache["sessions"][idx][direction]`, where
`direction` is None/all, -1/left or +1/right. `summarize_pc_psths` derives these
views from one full projection per session. Each cached view retains settings,
time centers, neuron IDs, model references, n_complete_bouts, n_excluded_bouts,
and per-mode/condition bout_ids, n_bouts, mean and sem. It drops bout_scores and
source recordings. Unknown directions enter only the all-bouts view; empty
subsets have zero counts and undefined mean/SEM.

The notebook groups original recording loads by source, releases the last
collection after precomputation, and validates a cache signature covering PCA
paths/hashes, selected sessions, baseline, extraction, binning, components and
smoothing. Changed settings require rerunning precomputation. Display limits
can change without recomputing within the cached interval. All summaries stay
in memory only; restarting the kernel requires precomputation again. Both
slider and direction callbacks only retrieve and redraw cached summaries.
