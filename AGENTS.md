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
- Notebook working directories can vary. Check the kernel's directory when resolving paths.

## Data files and alignment

Example triad in the workspace data directory

```text
FETA_hs_0_playback_0.005_data_ss
FETA_hs_0_playback_0.005_feature_ss
FETA_hs_0_playback_0.005_neuron_idx_ss
```

- All three are binary Python pickle files without filename extensions, loaded with
  `pickle.load()` from a file opened in `rb` mode.
- The animal is named in the prefix, for example FETA, HERCULE, MMELOIK, or NAPOLEON.
  Headstage is 0 or 1. Available headstages depend on the animal.
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
- `Block` carries block information. Its detailed meaning needs experiment-specific confirmation.
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