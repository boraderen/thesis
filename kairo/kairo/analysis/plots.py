import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from minisom import MiniSom
import pandas as pd
import numpy as np
import colorsys

def plot_activity_counts(stats: dict) -> go.Figure:
    counts = stats["activity_counts"]

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        labels={"x": "Count", "y": "Activity"},
        title="Activity Counts",
    )

    fig.update_yaxes(autorange="reversed")

    return fig

def plot_pca_variances(pca: PCA, cut_component: int = None) -> go.Figure:
    variances = pca.explained_variance_ratio_

    # without a cut the whole fitted pca is kept, with one only the first components
    kept = cut_component if cut_component else pca.n_components_

    fig = px.bar(
        x=list(range(1, len(variances) + 1)),
        y=variances,
        labels={"x": "Component", "y": "Explained variance"},
        title=(
            f"PCA reduces {pca.n_features_in_}D \u2192 {kept}D "
            f"({variances[:kept].sum():.1%} variance explained)"
        ),
    )

    # the line runs between the last kept component and the first dropped one
    if cut_component:
        fig.add_vline(x=cut_component + 0.5, line_dash="dash", line_color="darkred")

    return fig

def plot_som_u_matrix(som: MiniSom) -> go.Figure:
    # distance of every neuron to its neighbours: pale cells are the middle of a
    # cluster, dark ridges are the borders between two of them
    distances = som.distance_map()

    fig = px.imshow(
        distances,
        color_continuous_scale="Greys",
        labels={"x": "j", "y": "i", "color": "Distance"},
        title="SOM u-matrix",
    )

    return fig

def plot_som_heatmap(df: pd.DataFrame, som: MiniSom) -> go.Figure:
    # how many rows every neuron won, an empty cell is behaviour the map has room
    # for but the log never shows
    rows, cols, _ = som.get_weights().shape

    counts = df.groupby(["i", "j"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=range(rows), columns=range(cols), fill_value=0)

    fig = px.imshow(
        counts,
        color_continuous_scale="Blues",
        labels={"x": "j", "y": "i", "color": "Events"},
        title="Events per SOM cell",
    )

    return fig

def compute_som_color_mapping(size: tuple[int, int]) -> dict:
    # compute vfor each grid cell a very different color randomly
    # evenly spaced hues are as far apart as colors get, shuffling them keeps
    # neighbouring cells from ending up in similar tones
    cells = [(i, j) for i in range(size[0]) for j in range(size[1])]

    hues = np.linspace(0, 1, len(cells), endpoint=False)
    np.random.default_rng(0).shuffle(hues)

    colors = []

    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    return dict(zip(cells, colors))

def plot_som_colors(df: pd.DataFrame, color_mapping: dict) -> go.Figure:
    # plot the som grid with mapped colors
    # cells no row of df landed on stay grey
    rows = max(i for i, _ in color_mapping) + 1
    cols = max(j for _, j in color_mapping) + 1

    won = set(zip(df["i"], df["j"]))

    grid = np.full((rows, cols, 3), 235, dtype=np.uint8)

    for (i, j), color in color_mapping.items():
        if (i, j) in won:
            grid[i, j] = [int(color[k:k + 2], 16) for k in (1, 3, 5)]

    fig = px.imshow(
        grid,
        labels={"x": "j", "y": "i"},
        title="SOM cells",
    )

    return fig

def plot_case_trajectory(df: pd.DataFrame, color_mapping: dict, case_id: int) -> go.Figure:
    pass