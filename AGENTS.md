# eTheremin project context

These notes record discoveries and user decisions from exploratory work in November 2026.
Inspect the actual files and notebook settings when continuing an analysis. Examples below
describe inspected data and should not be assumed to apply to every animal or session.

## Working with the user

- Prefer direct access such as `features[session][column]` over extra intermediate feature tables.
- Use red for tracking and black for playback, including mean lines and shaded bands.
- Static Matplotlib plots work well. `%matplotlib widget` was slow and had widget-rendering
  problems in VS Code. The working event browser uses ipywidgets controls to redraw static
  figures when sliders are released (`continuous_update=False`).

## Layout and environment

- This directory is the Git root, nested inside the larger `etheremin/` workspace.
- Data files live in `../data/` relative to this repo, or `../../data/` from `notebooks/`.
- Use the shared `.venv/`, `pyproject.toml`, and `uv.lock` at this repo root for analyses.
  Dependencies are managed with uv. The Windows interpreter is `.venv/Scripts/python.exe`.
- `felicie/load_data.ipynb` and `felicie/utils_load_data.py` are Felicie's original material.
- `notebooks/explore_data.ipynb` contains the user's exploration and event-response analyses.
- `movement_bouts/analyse_movement_bouts.ipynb` contains the movement-bout analysis.
- `movement_bouts/perimotor_psth.ipynb` browses single-neuron onset-aligned PSTHs
  and tracking/playback rasters, using `psth.py`. It reads original recordings
  with saved movement onsets; extracts [-1.2, +1) s and displays [-0.5, +0.5) s.
  Subtract each bout/neuron's raw mean rate in [-1, -0.7) s, then smooth each
  corrected trace with an adjustable Gaussian sigma (0 disables; read the
  notebook's current `SMOOTHING_SIGMA_S`), using
  native 5 ms bins. Symmetric smoothing can spread changes across onset.
  Static plots show corrected mean firing rate +/- SEM across complete bouts, red for
  tracking and black for playback. Session/neuron and initial-direction controls
  redraw on change (sliders on release). One source collection is cached; only
  selected-neuron windows are copied. Bounds failures are excluded and counted.
  Conditions are assigned at onset; transition-crossing windows are not filtered.
  This notebook writes no derived data and does not use the pre-onset extraction.
  Its population section computes each neuron's condition PSTH across bouts,
  then averages neurons equally across all saved sessions by default. Population
  shading is SEM across recorded unit PSTHs, not bouts or independent animals.
  `POPULATION_SESSION_INDICES` and `POPULATION_DIRECTION` control its scope,
  independently of browser controls. Per-neuron traces and session counts remain
  in memory; source collections are loaded in turn. Units are session-specific.
  The neuron-type section reuses these traces and matches exact neuron IDs to
  `../data/df_clus_encoding_phd.pkl`: columns `neuron_idx`, `cluster_encoding`,
  `angle_rad`, `radius`. Mapping: Motor=0, Sensory=1, Sensorimotor=2, None=3.
  Four panels show equally weighted unit means +/- SEM with shared axes.
  Missing classifications are excluded and reported, never assigned to None.
  The inspected all-session population matches 1,257/1,401 units (345 Motor,
  126 Sensory, 376 Sensorimotor, 410 None; 144 unclassified). Classification IDs
  are unique; rerun coverage checks for changed session/data selections.
- Notebook working directories can vary. Check the kernel's directory when resolving paths.

## Neuron-type PCA notebook

- `movement_bouts/neuron_type_pca.ipynb` reuses the wider premotor extraction and
  the existing PCA/viewer/evaluation helpers. Its settings cell selects Motor,
  Sensory, Sensorimotor or None, `MIN_NEURONS` (default 5), and `ACTIVITY_MODE`
  (`raw` or `baseline_subtracted`, default corrected). Default analysis [-0.5, 0)
  uses five 100 ms windows; baseline is [-1, -0.7). No PSTH smoothing is applied.
- Match classification `neuron_idx` exactly to saved neuron IDs, retain recording
  neuron order, and filter neuron columns before fitting. Missing classifications
  never enter the None category. Preserve original session and bout indices.
- Sessions are filtered by selected-type count, not total unit count. The notebook
  exposes full `session_selection`, `selected_neuron_ids`, and condition-level
  `pca_fit_summary`; the default Sensorimotor cutoff retains 41 sessions/82 PCAs.
- Three independent interactive viewers color by initial position, final position,
  and initial direction (left blue/right red). Each uses separate condition PCAs.
  Position regressions and direction logistic regression pool all analysis windows
  per session/condition and show summary plus matched-condition score plots.
  These reuse descriptive in-sample methods, not significance tests or holdout.
  Starting position and direction have no displacement filter. Final-position
  regression defaults to no displacement filter; set
  `DESTINATION_MIN_DISPLACEMENT=5` for the earlier abs(end-start)>5 comparison.
- Results are saved under `data/derived/premotor_activity/runs/` with a
  `neuron_type_{type}_min{cutoff}_analysis_...` prefix, distinct from all-neuron
  runs. PCA bundles save exact selected IDs, eligibility, classification SHA-256,
  extraction provenance and preprocessing settings. Recoloring uses saved results.
  Change the settings and rerun below them to refit; tests in
  `movement_bouts/test_neuron_type_pca.py` check matching, cutoffs, neuron selection
  before PCA, and both activity modes using the notebook's actual code cells.

## Data files and alignment

Example triad in the workspace data directory

```text
FETA_hs_0_AC_playback_0.005_data_ss
FETA_hs_0_AC_playback_0.005_feature_ss
FETA_hs_0_AC_playback_0.005_neuron_idx_ss
```

- All three are binary Python pickle files without filename extensions, loaded with
  `pickle.load()` from a file opened in `rb` mode.
- The animal is named in the prefix, for example FETA, HERCULE, MMELOIK, or NAPOLEON.
  Headstage is 0 or 1. Available headstages depend on the animal.
- Filenames include the anatomical region after the headstage, using
  `{animal}_hs_{headstage}_{region}_playback_0.005_{suffix}`.
  The user confirmed that FETA headstage 1 and HERCULE headstage 1 are PMC.
  The remaining six animal/headstage collections use AC, matching Felicie's A1 list.
  All 24 local files were renamed in November 2026. Use the region-qualified names
  for all three files in a collection. The exploration notebook checks the selected
  region, file availability, session counts, and per-session alignment when loading.
- The user clarified that `playback` in these filenames does not identify the condition
  of every bin. Use the `Condition` feature to split tracking and playback.
- `0.005` specifies 5 ms bins (`dt = 0.005` seconds, 200 bins per second).
- Each triad contains a list of recording sessions for an animal/headstage combination.
  The directly inspected FETA headstage 0 triad has 12 entries in each list.
  The filenames identify a collection of sessions. The source of the 12-session selection
  and ordering across other collections has not been independently established.

| Loaded variable | Outer object | Entry at session index `s` |
| --- | --- | --- |
| `neural_data` | Python list | NumPy array, shape `(n_units, n_time_bins)`, inspected dtype `float64` |
| `features` | Python list | pandas DataFrame, shape `(n_time_bins, n_features)` |
| `neuron_ids` | Python list | Python list of `n_units` string identifiers |

- `neural_data[s][u, t]` aligns with `neuron_ids[s][u]` and `features[s].iloc[t]`.
- Unit counts and recording lengths vary across sessions. Check shapes instead of assuming
  a rectangular array across sessions.
- Inspected neural values are nonnegative integer-valued spike counts stored as floats.
  Divide by `dt` to convert counts per bin into spikes/s. The user identifies the units
  as WaveClus spike-sorted units.
- Feature rows have a RangeIndex of bin positions. Time from session start is `bin_index * dt`.
  Duration is `n_time_bins * dt`, and the final bin starts at `(n_time_bins - 1) * dt`.
- FETA headstage 0, session index 0 has 14 units and 482,730 bins, giving 2,413.65 seconds.
  Its feature table has shape `(482730, 17)` and there are 14 neuron IDs.
- Example identifier `FETA_20251204_SESSION_00_headstage_0_neuron0` encodes recording,
  headstage, and unit. These are session-specific IDs. Cross-session biological neuron
  identity and correspondence with original WaveClus cluster numbers need separate verification.
- File sizes range into hundreds of MB or GB. Inspect one animal/headstage collection at a time.

## Features and position reconstruction

The inspected tables contain these 17 columns

```text
Played_frequency, Condition, Block, Frequency_changes, Mock_frequency, Mock_change,
Change_direction, Mock_direction, Speed_x, Speed_x_signed, Sound_speed, Position,
Freq_position, Speed_bin, Acc_x_signed, Acc_x, Acc_sound
```

- `Played_frequency` gives the played sound frequency in Hz.
- `Condition` uses 0 for tracking and 1 for playback. Felicie's notebook labels -1 as
  warmup or silence. Analyses currently exclude other conditions from the two-condition comparison.
- `Frequency_changes` is a boolean event flag for changes in played frequency.
- `Mock_frequency` corresponds to the sound that would follow the animal's position during
  playback. `Mock_change` flags changes in that signal. Felicie's notebook describes
  `Mock_frequency` as 0 during tracking. Inspect values when needed.
- `Change_direction` and `Mock_direction` are derived from differences in the corresponding
  frequency signals, with values -1, 0, or 1 in the loading code.
- `Position` is reconstructed from frequency in Felicie's loading code. The code chooses
  played frequency for conditions 0 and -1, and mock frequency for playback. It interpolates
  sorted `unique_tones` onto equally spaced positions from 0 to 28 (commented as cm), then
  applies Gaussian smoothing with sigma 10 bins.
- `Freq_position` reconstructs position from played frequency in all conditions, with the
  same interpolation and smoothing. During playback, comparing it with `Position` compares
  the played-sound trajectory with the trajectory inferred from the mock signal.
- No separate `Mock_position` column was found in the inspected data. In playback,
  `Position` already uses mock frequency in the available loading implementation.
- Speed, acceleration, position, and direction fields are derived quantities. Their provenance
  comes from the available loading code. Verify physical units and scaling before quantitative
  interpretation, especially hard-coded speed factors in that code.
- These position fields already include preprocessing smoothing.


## Derived movement-bout data

- Save derived tables in the workspace's `data/derived/movement_bouts/` directory,
  which is `../data/derived/movement_bouts/` from the repository root and
  `../../data/derived/movement_bouts/` from `movement_bouts/`.
- `movement_bouts/analyse_movement_bouts.ipynb` loads all AC feature collections,
  calls `detect_movement_bouts()` from the adjacent `utils.py`, and exports the tables
  in its save cell immediately after detection. Rerun detection and saving together
  when changing extraction parameters.
- `movement_bouts/analyse_neuronal_activity.ipynb` reloads these derived tables for
  subsequent neural analysis. Run both notebooks with `movement_bouts/` as the kernel's
  working directory so helpers can be imported directly from `utils`.
- The user prefers pickle for lab-wide readability. Save DataFrames with
  `DataFrame.to_pickle()` and reload with `pd.read_pickle()`. Parameters use JSON.

| File | Contents |
| --- | --- |
| `movement_bouts.pkl` | One row per movement onset, including unfinished bouts |
| `immobility_bouts.pkl` | All detected immobility segments, including short segments, with a `qualifies` flag |
| `session_info.pkl` | Mapping from global session-entry index to animal, headstage, source session, feature file, and recording durations |
| `parameters.json` | `DT`, `SPEED_X_FACTOR`, `VELOCITY_THRESHOLD`, `MIN_IMMOBILITY_S`, Git commit, uncommitted-change flag, and UTC export time |

### Session identity and saved fields

- Both bout tables have an `idx` column linking to `session_info.loc[idx]` and the
  original flattened `features[idx]` list. Preserve the saved `session_info` index.
  Its `session` column is the zero-based session index within that animal/headstage
  collection, used to select the corresponding entry from its original pickle files.
- `session_info["feature_file"]` identifies the source collection. Replacing the
  `_feature_ss` suffix with `_data_ss` or `_neuron_idx_ss` identifies its aligned
  neural-data or neuron-ID file. Use the saved `session` to select the entry within it.
- Movement fields are `animal`, `headstage`, `session`, `condition`, `onset_bin`,
  `onset_time_s`, `end_bin`, `end_time_s`, `initial_position`, `initial_direction`,
  `final_position`, `total_movement`, `mean_speed`, and `idx`.
- Immobility fields are `start_bin`, `stop_bin`, `duration_s`, `qualifies`, `idx`,
  `animal`, `headstage`, and `session`.

## Derived premotor neuronal data

- `movement_bouts/utils.py` provides `extract_premotor_bouts()` for one session
  and `extract_all_premotor_bouts()` to load collections once and process their
  sessions individually. Pass `session_info.loc[[idx]]` to the latter for one session.
- `movement_bouts/analyse_neuronal_activity.ipynb` saves wider raw counts in
  `../data/derived/premotor_activity/extracted_-1.2_0/` relative to the repository
  root. The extraction interval is configurable; always read its saved metadata.
- `neuronal_bouts.pkl` has one `spike_counts` column. Its index is the original
  `movement_bouts` row index, never reset. Each cell holds an independent array
  shaped `(neurons, original time bins)` of spike counts. The new extraction is
  `[-1.2, 0)` seconds, giving 240 bins at 5 ms resolution, excluding movement onset.
  Superseded 100-bin extraction files at the premotor_activity root were removed
  after verifying every array matched the final 100 bins of the wider extraction,
  with identical bout indices, neuron IDs, exclusions, and source fingerprints.
  Load extraction tables and metadata from `extracted_-1.2_0/`.
- `movement_bouts` remains the reference for all behavioral metadata. Join neural
  rows by index; select a session before stacking arrays, since neuron counts vary.
- `neuron_ids_by_session.pkl` has a `neuron_ids` column indexed by session `idx`.
  `excluded_bouts.pkl` records skipped bout indices and exclusion reasons. Incomplete
  windows are skipped without failing the session; immobility is not checked again.
- `parameters.json` beside the extracted counts records DT, window_s, n_time_bins,
  source-table SHA-256 fingerprints,
  source bout-detection parameters, session indices, and export time. Regenerate
  neural outputs if the reference bout table is regenerated or its indices change.
- Averaging is separate from extraction: `N_BINS` divides ANALYSIS_WINDOW_S,
  initially [-0.5, 0), rather than the whole saved array. Five windows give 100 ms,
  two give 250 ms, one gives 500 ms. Sum counts in each window
  and divide by its duration for spikes/s; retain the original-resolution counts.
- `prepare_pca_input()` in `movement_bouts/utils.py` selects one session, averages
  its analysis-window counts, and returns two aligned in-memory DataFrames: `pca_input` (observations
  by neurons, in spikes/s) and `pca_labels` (original bout metadata plus `bin_number`
  and `bin_end_s`). Both use `(bout_id, bin_start_s)` as their index. Rows follow
  each bout's windows in chronological order. No extra intermediate files are saved.
- Pass the saved extraction_window_s and DT explicitly to prepare_pca_input, plus
  analysis_window_s, n_bins, activity_mode and (when used) baseline_window_s.
  Raw is the notebook default. In baseline_subtracted mode subtract each bout and
  neuron's baseline mean rate (initially [-1, -0.5)) from all its analysis windows.
  Units stay spikes/s; negatives are meaningful. The two-table return and original
  observation IDs are unchanged. Windows must align to DT, fit the saved array,
  and baseline must end at/before analysis start. Do not recheck immobility.
- `fit_pca_by_condition()` fits separate PCAs for tracking (0) and playback (1).
  Each centers its own firing rates, without variance standardization or whitening,
  and retains all components. Each condition entry holds `model` and `results`
  (coordinates joined to bout/window labels). Conditions with fewer than two
  observations are omitted. `plot_pca_session()` only draws these saved results.
- Notebook section 7 computes all sessions and saves models and labeled results
  together under the selected run_dir/pca/pca_{N_BINS}bins.pkl. Constants define
  run_dir = premotor_dir/runs/analysis_{start}_{end}_{N_BINS}bins_{activity_mode},
  adding baseline bounds for baseline_subtracted. All viewers, projections and
  evaluations follow central pca_path/evaluation_dir variables. Prior root-level
  results are retained. Load with `pd.read_pickle()`. This dictionary contains
  `sessions[idx][condition]`, `settings`, `session_info`, `skipped_sessions`, and
  global `position_limits`. The current 5-bin file has 160 models for 80 sessions.
- PCA settings record activity_mode, analysis_window_s, baseline_window_s (None
  for raw), extraction_directory and embedded extraction_parameters. Projection
  inputs must be prepared from the reference PCA's saved settings, not the current
  notebook preprocessing constants. Section 3 includes a raw/corrected example
  heatmap and baseline rates; corrected heatmaps use a diverging scale centered
  at zero. Extraction and analysis interval changes must not be conflated.
- To browse saved PCAs, run sections 1, 8, and 9. Section 8 displays a position
  viewer; section 9 displays a direction viewer. Each has its own session slider
  and one Plotly FigureWidget, created by `make_interactive_pca_browser()` with
  `color_by="position"` or `color_by="direction"`. Each viewer has two 3D panels
  (tracking/playback) with cameras linked only within that viewer. Session and
  camera controls are independent between viewers. All points are circles;
  direction colors are blue left, red right, gray unknown. Position limits are
  fixed across sessions. Sliders update existing traces without preparation or
  fitting. Dependencies plotly/anywidget are managed by uv. Section 6 remains
  a static Matplotlib preview.

- User explicitly selected blue for left and red for right in both PCA direction
  panels; this overrides the earlier reservation of red for tracking in these panels.

- Section 9 is the direction-colored viewer with starting-position
  filtering: all, left (<14), or right (>=14). The midpoint defaults to 14 on the
  reconstructed 0?28 scale and is adjustable. `filter_position=True` enables the
  selector. Filtering hides saved observations without refitting; full-session axis
  ranges and linked cameras are preserved, including for empty selections.

- Section 10 adds `make_bout_pca_browser()`: an independent session slider and
  bout slider (updates on release). One bout is highlighted in its own condition panel, with labeled
  time bins, a light-to-dark time scale and chronological connecting line. Other
  bouts retain faint starting-position colors (fixed Viridis scale); full-session axis ranges and linked cameras are preserved.
  Shared camera synchronization lives in `_link_pca_cameras()`. No refitting occurs.

- Section 11 projects playback onto each session's saved tracking PCA using
  `project_onto_tracking_pca()`. Both conditions use tracking mean/loadings via
  `transform()`; no refitting or playback recentering. Results are saved separately
  in the selected run's `pca/pca_tracking_projection_{N_BINS}bins.pkl` with `projection_basis="tracking"`.
  The new position-colored viewer has matching axis limits and linked cameras;
  percentages refer to tracking variance. Existing separate-PCAs caches are unchanged.

- Plotly FigureWidget 7.1 can draw successfully but emit repeated ValueErrors in
  `_remove_overlapping_props` when browser trace-default acknowledgements contain
  typed NumPy arrays. Pass ordinary lists for coordinates, marker colors, text and
  customdata at the interactive plotting boundary. Section 10 exposed this issue;
  regression checks must replay `_js2py_traceDeltas`, not just camera callbacks.

- A populated FigureWidget nested in a VBox can still serialize zero traces:
  VBox does not invoke the child's `_repr_mimebundle_()` display hook. Call that
  hook once after populating the initial session and before returning the container.
  It synchronizes Plotly's _widget_data/_widget_layout with the actual data/layout.
  This applies to the PCA, bout-trajectory, and endpoint-browser factories. Tests
  in test_pca_browser.py rebuild figures from get_state(), exercise slider updates,
  and replay frontend trace/camera events. Constructing the widget without checking
  its serialized state does not verify that an embedded figure will display.

- Section 12 adds a destination-colored viewer using `color_by="destination"`.
  Colors use saved `final_position` (observed movement endpoint), constant across
  each bout's windows. Unfinished bouts with missing endpoints are shown in gray.
  The fixed positional scale matches the starting-position view; no refitting.

- Section 13 (`make_start_end_position_browser`) compares start and final position
  per session, one point per completed bout, split tracking/playback. It displays
  identity lines, Pearson r, median absolute net displacement and empirical CDFs
  of |end-start|. Missing endpoints are excluded and counted. Use position units
  without assuming cm; net displacement can be small after a return movement.

- Section 14 displays destination-colored PCA with `filter_displacement=True`: an
  adjustable minimum |final-initial| cutoff, initially 2 position units. Retain >=
  cutoff, exclude missing endpoints, and filter all windows of each bout together.
  This hides points without refitting; full-session axis ranges stay fixed.

## Shared representation/evaluation interface

- `movement_bouts/ANALYSIS_DATA_FORMAT.md` documents the common observation format
  for neuronal responses, PCA, future UMAP analyses and metrics. Read it before
  adding representations or evaluation functions.
- Evaluation helpers belong in `movement_bouts/evaluation.py`, separate from
  extraction, model fitting and plotting in `utils.py`. The new module currently
  provides `validate_data_labels()`, `fit_label_regression()`, and
  `plot_label_regression()`. Silhouette and randomization remain to be added.
- Use aligned numeric `data` and metadata `labels` DataFrames with identical unique
  `(bout_id, bin_start_s)` indices and row order. Select needed coordinate and
  target columns directly from saved results; no splitting/conversion helper is
  required. Preserve observation IDs when filtering. Missing targets are handled
  per metric, not removed globally.

- Prefer direct column selection over extra adapter layers. The user asked to remove
  the unnecessary coordinate/label splitting helper: the notebook already separates
  PCA inputs and labels. Add abstractions for concrete repeated work, not speculative
  future reuse.

- Section 15 demonstrates descriptive ordinary linear regression for any numeric
  continuous label using explicitly selected features/coordinates. Default: session
  0, selected PCs, initial_position, last window; tracking/playback fitted separately.
  `fit_label_regression()` returns model, target, in-sample R2/MAE, n_excluded,
  and indexed observed/fitted/residual values. Exclude only missing/nonfinite target
  rows; reject insufficient/constant/nonnumeric targets. No held-out evaluation,
  scaling, PCA refitting or randomization. Plotting is separate. Unit checks live
  in `movement_bouts/test_evaluation.py` and run with unittest discovery.

- Section 16 repeats section 15 regression with `final_position`, reusing the
  session, PC selection and window settings. Starting-position results remain
  separate. No displacement filter is applied; missing endpoints are excluded
  by the generic helper. Both analyses are descriptive/in-sample.

- Section 17 repeats destination regression for `abs(final_position -
  initial_position) > 5`, including both directions and preserving PC/window
  settings. Missing endpoints are excluded. The earlier signed >8 filter was
  corrected to match the goal of excluding bouts ending close to their start.

- Section 18 runs `evaluation.run_label_regressions()` once per session and
  condition, pooling all premotor windows. The user explicitly rejected the
  additional per-window batch analysis; do not reintroduce it without discussion.
  Default: final_position from PC1-PC3, abs(end-start)>5. Summary and fit keys are
  (idx, condition); prediction tables retain all (bout_id, bin_start_s) rows.
  Report n_observations and n_bouts separately. Residual degrees of freedom and
  adjusted R2 use design-row count and do not correct within-bout dependence.
  Invalid subsets have status/reason; saturated fits have undefined adjusted R2.
- Batch bundles are saved under the selected run's `evaluation/` directory with
  `_pooled.pkl` suffix and `settings["window_mode"]="pooled"`. Models, predictions
  and settings are saved together. `plot_regression_summary()` displays one dot
  per PCA against session idx, with separate tracking/playback panels and mean
  lines. It rejects the superseded per-window summary format. No fitting occurs
  in plotting. Sections 15-17 remain the earlier single-window examples.

- Section 19 (`plot_paired_regressions`) matches tracking/playback scores by
  session idx and plots paired values plus tracking-minus-playback differences.
  It returns the figure, axes and paired table, including complete finite pairs
  with status ok only. Default `PAIR_SCORE="r2_adjusted"`; use "r2" for ordinary R2.
  It reads saved scores without fitting or adding significance tests.

- Section 20 repeats the pooled batch and paired plots for initial_position,
  reusing section 18 source PCs and feature columns, with NO displacement filter.
  All bouts with finite initial_position are included, regardless of destination
  or movement size. Saves a separate all_bouts initial-position bundle; destination
  fits retain their existing filter. The user explicitly removed this filter from
  the starting-position analysis.

- Section 21 compares existing starting-position regression scores with side
  preference and position spread, without refitting. Helpers in evaluation.py:
  `summarize_starting_position_bias`, `plot_position_bias_scores`, and
  `plot_paired_bias_differences`. Use the actual fitted observations and count
  each bout once. Midpoint 14: left <14, right >=14; signed bias is
  (n_right-n_left)/n_bouts, absolute bias ranges from 0 balanced to 1 one-sided.
  Store sample standard deviation and conventional skewness as distinct measures.
  The four-panel plot compares adjusted R2 (or ordinary R2) with absolute bias and
  spread by condition; the paired plot compares tracking-minus-playback differences
  in score and absolute bias, matched by idx. These are descriptive checks, not
  significance tests or corrections for behavioral sampling. The summary is saved
  beside its starting-position regression bundle with `_position_bias.pkl` suffix.

- Section 22 fits one descriptive left/right logistic regression per session and
  condition, using PC1-PC3 by default, all windows pooled, no displacement filter.
  `fit_direction_logistic` and `run_direction_logistics` live in evaluation.py.
  Target initial_direction -1/+1 is movement direction, not starting side; exclude
  0/missing/unknown labels and skip subsets without both directions. Use a saved
  StandardScaler + L2 logistic Pipeline (C=1, intercept, balanced class weights).
  Standardize selected coordinates only within the classifier; do not refit PCA.
  Save models, indexed direction/probability predictions, summaries and settings
  under evaluation/logistic_initial_direction_..._pooled.pkl. Summary counts
  distinguish observations and bouts, including both directions separately.
  Existing plot_regression_summary/plot_paired_regressions support
  score="balanced_accuracy", with 0.5 absolute-score and zero difference references.
  These remain in-sample descriptive scores; no holdout, randomization or
  significance tests were requested. Sections 18-21 keep their prior behavior.
