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
- Notebook working directories can vary. Check the kernel's directory when resolving paths.

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
- `movement_bouts/analyse_neuronal_activity.ipynb` extracts and saves the tables
  in `../data/derived/premotor_activity/` relative to the repository root.
- `neuronal_bouts.pkl` has one `spike_counts` column. Its index is the original
  `movement_bouts` row index, never reset. Each cell holds an independent array
  shaped `(neurons, original time bins)` of spike counts. The current window is
  `[-0.5, 0)` seconds, giving 100 bins at 5 ms resolution, excluding movement onset.
- `movement_bouts` remains the reference for all behavioral metadata. Join neural
  rows by index; select a session before stacking arrays, since neuron counts vary.
- `neuron_ids_by_session.pkl` has a `neuron_ids` column indexed by session `idx`.
  `excluded_bouts.pkl` records skipped bout indices and exclusion reasons. Incomplete
  windows are skipped without failing the session; immobility is not checked again.
- `parameters.json` records extraction timing, source-table SHA-256 fingerprints,
  source bout-detection parameters, session indices, and export time. Regenerate
  neural outputs if the reference bout table is regenerated or its indices change.
- Averaging is separate from extraction: `N_BINS` is the number of equal temporal
  windows (5 gives 100 ms, 2 gives 250 ms, 1 gives 500 ms). Sum counts in each window
  and divide by its duration for spikes/s; retain the original-resolution counts.
- `prepare_pca_input()` in `movement_bouts/utils.py` selects one session, averages
  its counts, and returns two aligned in-memory DataFrames: `pca_input` (observations
  by neurons, in spikes/s) and `pca_labels` (original bout metadata plus `bin_number`
  and `bin_end_s`). Both use `(bout_id, bin_start_s)` as their index. Rows follow
  each bout's windows in chronological order. No extra intermediate files are saved.
- `fit_pca_by_condition()` fits separate PCAs for tracking (0) and playback (1).
  Each centers its own firing rates, without variance standardization or whitening,
  and retains all components. Each condition entry holds `model` and `results`
  (coordinates joined to bout/window labels). Conditions with fewer than two
  observations are omitted. `plot_pca_session()` only draws these saved results.
- Notebook section 7 computes all sessions and saves models and labeled results
  together in `data/derived/premotor_activity/pca/pca_{N_BINS}bins.pkl` relative to
  the workspace root. Load with `pd.read_pickle()`. This dictionary contains
  `sessions[idx][condition]`, `settings`, `session_info`, `skipped_sessions`, and
  global `position_limits`. The current 5-bin file has 160 models for 80 sessions.
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
  in `pca/pca_tracking_projection_{N_BINS}bins.pkl` with `projection_basis="tracking"`.
  The new position-colored viewer has matching axis limits and linked cameras;
  percentages refer to tracking variance. Existing separate-PCAs caches are unchanged.

- Plotly FigureWidget 7.1 can draw successfully but emit repeated ValueErrors in
  `_remove_overlapping_props` when browser trace-default acknowledgements contain
  typed NumPy arrays. Pass ordinary lists for coordinates, marker colors, text and
  customdata at the interactive plotting boundary. Section 10 exposed this issue;
  regression checks must replay `_js2py_traceDeltas`, not just camera callbacks.
