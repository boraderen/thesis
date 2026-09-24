import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from minisom import MiniSom
import pandas as pd
import numpy as np
import colorsys
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from . import (
    META,
    _cluster_name,
    get_som_case_trajectory,
    get_kmeans_case_trajectory,
    get_dbscan_case_trajectory,
    get_kmeans_state_distances,
    get_dbscan_state_distances,
)

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

    fig.update_layout(xaxis_dtick=1, yaxis_dtick=1)

    return fig

def plot_som_heatmap(df: pd.DataFrame, size: tuple[int, int], start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> go.Figure:
    # how many rows every neuron won, an empty cell is behaviour the map has room
    # for but the log never shows
    df = df[["time:timestamp", "i", "j"]]

    if start_date:
        df = df[df["time:timestamp"] >= start_date]

    if end_date:
        df = df[df["time:timestamp"] <= end_date]

    # every cell of the map is drawn, also the ones no row landed on in this range,
    # so the plots of different date ranges can be compared
    counts = df.groupby(["i", "j"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=range(size[0]), columns=range(size[1]), fill_value=0)

    # on grids bigger than 10x10 the cells get too small to read a number in them
    show_counts = size[0] <= 10 and size[1] <= 10

    fig = px.imshow(
        counts,
        text_auto=",d" if show_counts else False,
        color_continuous_scale="Blues",
        labels={"x": "j", "y": "i", "color": "Events"},
        title=f"Events per SOM cell ({start_date or 'start'} to {end_date or 'end'})",
    )

    fig.update_layout(xaxis_dtick=1, yaxis_dtick=1)

    return fig

def compute_som_color_mapping(size: tuple[int, int]) -> dict:
    # compute vfor each grid cell a very different color randomly
    cells = [(i, j) for i in range(size[0]) for j in range(size[1])]

    return dict(zip(cells, _distinct_colors(len(cells))))

def plot_som_colors(df: pd.DataFrame, color_mapping: dict) -> go.Figure:
    # plot the som grid with mapped colors, and how many rows landed on every cell
    counts = df[["i", "j"]].value_counts()

    rows = max(i for i, _ in color_mapping) + 1
    cols = max(j for _, j in color_mapping) + 1

    # every cell gets its own number, and a colorscale with one stop per number
    # paints it in its mapped color. drawn like this as a heatmap, the counts shrink
    # with the cells and switch between dark and light text as in plot_som_heatmap
    index = np.zeros((rows, cols))
    text = np.empty((rows, cols), dtype=object)
    scale = []

    for k, ((i, j), color) in enumerate(color_mapping.items()):
        index[i, j] = k
        text[i, j] = f"{counts.get((i, j), 0):,}"
        scale.append([k / (len(color_mapping) - 1), color])

    fig = px.imshow(
        index,
        color_continuous_scale=scale,
        labels={"x": "j", "y": "i"},
        title="SOM state colors",
    )

    fig.update_layout(xaxis_dtick=1, yaxis_dtick=1, coloraxis_showscale=False)

    # hovering a cell names its state and count instead of its number
    fig.update_traces(text=text, hovertemplate="State (%{y}, %{x})<br>%{text} events<extra></extra>")

    # on grids bigger than 10x10 the cells get too small to read a number in them
    if rows <= 10 and cols <= 10:
        fig.update_traces(texttemplate="%{text}")

    return fig

def plot_som_case_trajectory(df: pd.DataFrame, color_mapping: dict, case_id: str) -> go.Figure:
    colors = {f"({i}, {j})": color for (i, j), color in color_mapping.items()}

    return _plot_state_timeline(get_som_case_trajectory(df, case_id), colors, case_id)

def compute_kmeans_color_mapping(k: int) -> dict:
    # a very different color for every cluster
    return dict(zip(range(k), _distinct_colors(k)))

def compute_dbscan_color_mapping(dbscan: DBSCAN) -> dict:
    # a very different color for every cluster, and grey for the noise
    clusters = range(dbscan.labels_.max() + 1)

    color_mapping = dict(zip(clusters, _distinct_colors(len(clusters))))
    color_mapping[-1] = "#bbbbbb"

    return color_mapping

def plot_kmeans_colors(df: pd.DataFrame, color_mapping: dict) -> go.Figure:
    return _plot_cluster_colors(df, color_mapping, "K-means cluster colors")

def plot_dbscan_colors(df: pd.DataFrame, color_mapping: dict) -> go.Figure:
    return _plot_cluster_colors(df, color_mapping, "DBSCAN cluster colors")

def plot_kmeans_frequencies(df: pd.DataFrame, color_mapping: dict, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> go.Figure:
    return _plot_cluster_frequencies(df, color_mapping, start_date, end_date, "Events per k-means cluster")

def plot_dbscan_frequencies(df: pd.DataFrame, color_mapping: dict, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> go.Figure:
    return _plot_cluster_frequencies(df, color_mapping, start_date, end_date, "Events per DBSCAN cluster")

def plot_kmeans_distances(kmeans: KMeans) -> go.Figure:
    # how far apart the cluster centers are, close clusters are similar states
    return _plot_state_distances(get_kmeans_state_distances(kmeans), "Distances between k-means clusters")

def plot_dbscan_distances(dbscan: DBSCAN) -> go.Figure:
    # dbscan has no centers, see get_dbscan_state_distances for what stands in for them
    return _plot_state_distances(get_dbscan_state_distances(dbscan), "Distances between DBSCAN clusters")

def plot_dbscan_k_distance(df: pd.DataFrame, min_samples: int = 5, distance: str = "euclidean") -> go.Figure:
    # distance of every row to its k-th nearest row, with k = min_samples and the row
    # itself counted as in dbscan, sorted. rows left of the knee sit in dense regions,
    # rows right of it are outliers, so the distance at the knee is a good eps
    data = df.drop(columns=META, errors="ignore").to_numpy()

    neighbours = NearestNeighbors(n_neighbors=min_samples, metric=distance).fit(data)
    distances, _ = neighbours.kneighbors(data)

    fig = px.line(
        y=np.sort(distances[:, -1]),
        render_mode="webgl",
        labels={"x": "Rows sorted by k-distance", "y": "k-distance"},
        title=f"DBSCAN k-distance curve (k = min_samples = {min_samples})",
    )

    return fig

def plot_kmeans_case_trajectory(df: pd.DataFrame, color_mapping: dict, case_id: str) -> go.Figure:
    colors = {_cluster_name(c): color for c, color in color_mapping.items()}

    return _plot_state_timeline(get_kmeans_case_trajectory(df, case_id), colors, case_id)

def plot_dbscan_case_trajectory(df: pd.DataFrame, color_mapping: dict, case_id: str) -> go.Figure:
    colors = {_cluster_name(c): color for c, color in color_mapping.items()}

    return _plot_state_timeline(get_dbscan_case_trajectory(df, case_id), colors, case_id)


# shared by the functions above

def _distinct_colors(n: int) -> list:
    # evenly spaced hues are as far apart as colors get, shuffling them keeps
    # neighbouring cells from ending up in similar tones
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.default_rng(0).shuffle(hues)

    colors = []

    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    return colors

def _plot_cluster_colors(df: pd.DataFrame, color_mapping: dict, title: str) -> go.Figure:
    # one tile per cluster in its mapped color, with how many rows landed in it.
    # drawn as a heatmap the same way as plot_som_colors, so the counts shrink with
    # the tiles and switch between dark and light text
    counts = df["cluster"].value_counts()
    clusters = list(color_mapping)

    fig = px.imshow(
        [list(range(len(clusters)))],
        x=[_cluster_name(c) for c in clusters],
        color_continuous_scale=list(color_mapping.values()),
        labels={"x": "Cluster"},
        title=title,
    )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(visible=False)

    text = [[f"{counts.get(c, 0):,}" for c in clusters]]
    fig.update_traces(text=text, hovertemplate="Cluster %{x}<br>%{text} events<extra></extra>")

    # with more than 10 clusters the tiles get too small to read a number in them
    if len(clusters) <= 10:
        fig.update_traces(texttemplate="%{text}")

    return fig

def _plot_cluster_frequencies(df: pd.DataFrame, color_mapping: dict, start_date, end_date, title: str) -> go.Figure:
    df = df[["time:timestamp", "cluster"]]

    if start_date:
        df = df[df["time:timestamp"] >= start_date]

    if end_date:
        df = df[df["time:timestamp"] <= end_date]

    # every cluster is drawn, also the ones no row landed on in this range,
    # so the plots of different date ranges can be compared
    clusters = list(color_mapping)
    counts = df["cluster"].value_counts().reindex(clusters, fill_value=0)
    names = [_cluster_name(c) for c in clusters]

    fig = px.bar(
        x=names,
        y=counts.to_numpy(),
        color=names,
        color_discrete_map={_cluster_name(c): color for c, color in color_mapping.items()},
        text_auto=",d" if len(clusters) <= 10 else False,
        labels={"x": "Cluster", "y": "Events"},
        title=f"{title} ({start_date or 'start'} to {end_date or 'end'})",
    )

    fig.update_layout(showlegend=False)

    return fig

def _plot_state_distances(distances: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        distances,
        text_auto=".2f" if len(distances) <= 10 else False,
        color_continuous_scale="Greys",
        labels={"x": "Cluster", "y": "Cluster", "color": "Distance"},
        title=title,
    )

    return fig

def _plot_state_timeline(visits: pd.DataFrame, colors: dict, case_id: str) -> go.Figure:
    # one bar per visit of a state, from when the case entered it until it left.
    # the case id is the same for every bar, using it as y keeps them in one lane
    fig = px.timeline(
        visits,
        x_start="start",
        x_end="end",
        y=[str(case_id)] * len(visits),
        color="state",
        color_discrete_map=colors,
        hover_data=["events"],
        title=f"Case {case_id}",
    )

    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)

    return fig
