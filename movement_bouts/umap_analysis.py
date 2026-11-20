"""Independent condition UMAP fits and the raw/baseline comparison figure."""

import numpy as np
import pandas as pd

from evaluation import validate_data_labels


def fit_umap_by_condition(data, labels, *, n_neighbors=30, min_dist=0.1, random_state=42):
    """Embed neural rates directly, without PCA, scaling or behavioral targets.

    Small conditions are explicitly skipped rather than silently changing k.
    Results retain the original bout/window observation index and all metadata.
    """
    from umap import UMAP

    validate_data_labels(data, labels)
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 2:
        raise ValueError("n_neighbors must be an integer >=2.")
    fits, rows = {}, []
    for condition in (0, 1):
        subset = data.loc[labels["condition"].eq(condition)]
        row = {"condition": condition, "n_observations": len(subset),
               "n_bouts": subset.index.get_level_values("bout_id").nunique(),
               "n_neurons": data.shape[1], "n_neighbors": n_neighbors,
               "status": "skipped", "reason": ""}
        if len(subset) <= n_neighbors:
            row["reason"] = f"Need more than {n_neighbors} observations; lower n_neighbors."
        else:
            model = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric="euclidean", random_state=random_state, n_jobs=1,
                         init="random", compatibility_layout=True)
            coordinates = pd.DataFrame(model.fit_transform(subset), index=subset.index,
                                       columns=["UMAP1", "UMAP2", "UMAP3"])
            if not np.isfinite(coordinates.to_numpy()).all():
                raise ValueError(f"Nonfinite UMAP coordinates for condition {condition}.")
            fits[condition] = {"model": model, "results": labels.loc[subset.index].join(coordinates),
                               "neuron_ids": data.columns.tolist()}
            row["status"] = "ok"
        rows.append(row)
    return fits, pd.DataFrame(rows).set_index("condition")


DIRECTION_COLORS = {-1: "#0000ff", 1: "#ff0000", 0: "#8c8c8c"}


def _color_groups(results, color_by):
    """Partition a saved panel into color groups, retaining unknown labels."""
    if color_by == "position":
        positions = pd.to_numeric(results["initial_position"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(positions)
        return [(results.iloc[np.flatnonzero(finite)], positions[finite].tolist(), "Starting position"),
                (results.iloc[np.flatnonzero(~finite)], "#8c8c8c", "Unknown position")]
    if color_by == "direction":
        directions = results["initial_direction"]
        return [(results.loc[mask.fillna(False)], DIRECTION_COLORS[direction], name)
                for mask, direction, name in ((directions.eq(-1), -1, "Leftward"),
                                               (directions.eq(1), 1, "Rightward"),
                                               (~directions.isin([-1, 1]), 0, "Unknown direction"))]
    raise ValueError("color_by must be 'position' or 'direction'.")


def plot_umap_comparison(bundle, *, color_by="position"):
    """Draw one annotation per figure from saved embeddings, without fitting."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    if color_by not in ("position", "direction"):
        raise ValueError("color_by must be 'position' or 'direction'.")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained",
                             subplot_kw={"projection": "3d"})
    norm = Normalize(*bundle["position_limits"])
    cmap = plt.get_cmap("viridis")
    for row, mode in enumerate(("raw", "baseline_subtracted")):
        for col, condition in enumerate((0, 1)):
            ax = axes[row, col]
            fit = bundle["fits"][mode].get(condition)
            name = "Tracking" if condition == 0 else "Playback"
            mode_label = "Raw" if mode == "raw" else "Baseline-subtracted"
            ax.set(xlabel="UMAP 1", ylabel="UMAP 2", zlabel="UMAP 3", title=f"{mode_label} | {name}")
            if fit is None:
                reason = bundle["fit_summary"].loc[(mode, condition), "reason"]
                ax.text2D(.5, .5, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
                continue
            results = fit["results"]
            for points, color, label in _color_groups(results, color_by):
                colors = cmap(norm(color)) if isinstance(color, list) else color
                ax.scatter(points["UMAP1"], points["UMAP2"], points["UMAP3"], c=colors, marker="o",
                           s=40, alpha=.85, edgecolors=".25", linewidths=.35, depthshade=False)
            n_bouts = results.index.get_level_values("bout_id").nunique()
            ax.set_title(f"{mode_label} | {name}\n{len(results):,} windows / {n_bouts:,} bouts")
    if color_by == "position":
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                     label="Starting position (position units)", shrink=.8)
    else:
        axes[0, 0].legend(handles=[Line2D([], [], color=DIRECTION_COLORS[d], marker="o", linestyle="", label=l)
                                  for d, l in ((-1, "Leftward"), (1, "Rightward"), (0, "Unknown direction"))],
                          loc="best", fontsize=8)
    settings = bundle["settings"]
    label = "Starting position" if color_by == "position" else "Movement direction"
    fig.suptitle(f"{label} | Session {settings['idx']} | neighbors={settings['n_neighbors']} | seed={settings['random_state']}\n"
                 "Separate fits: panel locations and axes are independent")
    return fig, axes


def make_interactive_umap_comparison(bundle, *, color_by="position", point_size=5):
    """Independent position/direction views of exactly the same saved coordinates."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if color_by not in ("position", "direction"):
        raise ValueError("color_by must be 'position' or 'direction'.")
    titles = [f"{mode} | {condition}" for mode in ("Raw", "Baseline-subtracted")
              for condition in ("Tracking", "Playback")]
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}]*2]*2,
                        subplot_titles=titles, horizontal_spacing=.05, vertical_spacing=.1)
    for row, mode in enumerate(("raw", "baseline_subtracted"), 1):
        for col, condition in enumerate((0, 1), 1):
            fit = bundle["fits"][mode].get(condition)
            panel = (row-1)*2 + col-1
            if fit is None:
                reason = bundle["fit_summary"].loc[(mode, condition), "reason"]
                fig.layout.annotations[panel].text += f"<br>{reason}"
                continue
            results = fit["results"]
            n_bouts = results.index.get_level_values("bout_id").nunique()
            fig.layout.annotations[panel].text += f"<br>{len(results):,} windows / {n_bouts:,} bouts"
            for points, color, name in _color_groups(results, color_by):
                marker = {"size": point_size, "symbol": "circle", "opacity": .85,
                          "line": {"color": "#404040", "width": .6}, "color": color}
                if isinstance(color, list):
                    marker["coloraxis"] = "coloraxis"
                fig.add_trace(go.Scatter3d(
                    x=points["UMAP1"].tolist(), y=points["UMAP2"].tolist(),
                    z=points["UMAP3"].tolist(), mode="markers", marker=marker,
                    name=name, legendgroup=name, showlegend=(row == col == 1 and color_by == "direction"),
                    text=[f"Bout {bout} | window {start:g} s | start position {pos} | direction {direction}"
                          for (bout, start), pos, direction in zip(points.index, points["initial_position"], points["initial_direction"])],
                    hovertemplate="%{text}<extra>%{fullData.name}</extra>"), row=row, col=col)
    settings = bundle["settings"]
    label = "Starting position" if color_by == "position" else "Movement direction"
    fig.update_layout(height=1000, margin={"l": 10, "r": 10, "b": 10, "t": 100},
                      title=f"{label} | Session {settings['idx']} | neighbors={settings['n_neighbors']} | seed={settings['random_state']}"
                            "<br><sup>Drag to rotate; scroll to zoom. Each panel has independent axes.</sup>",
                      legend={"orientation": "h", "y": -.02})
    if color_by == "position":
        fig.update_layout(coloraxis={"colorscale": "Viridis", "cmin": bundle["position_limits"][0],
                                     "cmax": bundle["position_limits"][1],
                                     "colorbar": {"title": "Starting position"}})
    fig.update_scenes(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3", aspectmode="cube")
    return fig
