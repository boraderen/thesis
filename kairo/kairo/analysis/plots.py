import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from . import (
    META,
    DIVERGENCES,
    REFERENCES,
    get_som_state_distances,
    get_kmeans_state_distances,
    get_dbscan_state_distances,
    _cluster_name,
    _unit,
)

# dbscan noise is no state, it gets a grey lighter than every state color
NOISE_COLOR = "#e4e4e4"


# log

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


# PCA

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

def plot_pca_reconstruction_errors(errors: pd.Series, cut_component: int = None) -> go.Figure:
    # errors is what compute_pca_reconstruction_errors returns. the elbow of the curve, where
    # one more component barely lowers the error, is a good cut
    if cut_component:
        title = f"PCA reconstruction error with {cut_component} components kept: {errors.loc[cut_component]:,.3f}"
    else:
        title = "PCA reconstruction error per number of components kept"

    fig = px.line(
        x=errors.index,
        y=errors.to_numpy(),
        markers=True,
        labels={"x": "Components kept", "y": "Reconstruction error"},
        title=title,
    )

    # the line runs through the error of the kept components
    if cut_component:
        fig.add_vline(x=cut_component, line_dash="dash", line_color="darkred")

    return fig


# SOM

def get_som_state_colors(som: MiniSom, distance: str = "euclidean") -> dict:
    # a color for every cell of the grid, close cells by the distance of their neurons get
    # close colors, see _colors_from_distances
    distances = get_som_state_distances(som, distance)
    rows, cols, _ = som.get_weights().shape
    cells = [(i, j) for i in range(rows) for j in range(cols)]

    return dict(zip(cells, _colors_from_distances(distances.to_numpy())))

def plot_som_state_colors(df: pd.DataFrame, state_colors: dict) -> go.Figure:
    # plot the som grid with the colors of its states, and how many rows landed on every cell
    counts = df[["i", "j"]].value_counts()
    unit = _unit(df)

    rows = max(i for i, _ in state_colors) + 1
    cols = max(j for _, j in state_colors) + 1

    index = np.zeros((rows, cols))
    text = np.empty((rows, cols), dtype=object)
    scale = []

    for k, ((i, j), color) in enumerate(state_colors.items()):
        index[i, j] = k
        text[i, j] = f"{counts.get((i, j), 0):,}"
        scale.append([k / max(len(state_colors) - 1, 1), color])

    fig = px.imshow(
        index,
        color_continuous_scale=scale,
        labels={"x": "j", "y": "i"},
        title="SOM state colors",
    )

    fig.update_layout(xaxis_dtick=1, yaxis_dtick=1, coloraxis_showscale=False)

    fig.update_traces(text=text, hovertemplate=f"State (%{{y}}, %{{x}})<br>%{{text}} {unit}<extra></extra>")

    if rows <= 10 and cols <= 10:
        fig.update_traces(texttemplate="%{text}")

    return fig

def plot_som_heatmap(df: pd.DataFrame, size: tuple[int, int], start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> go.Figure:
    # how many rows every neuron won, an empty cell is behaviour the map has room
    # for but the log never shows
    unit = _unit(df).capitalize()
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
        labels={"x": "j", "y": "i", "color": unit},
        title=f"{unit} per SOM cell ({start_date or 'start'} to {end_date or 'end'})",
    )

    fig.update_layout(xaxis_dtick=1, yaxis_dtick=1)

    return fig

def plot_som_u_matrix(som: MiniSom) -> go.Figure:
    # mean distance of every neuron to its up to eight neighbours, divided by the largest one:
    # pale cells are the middle of a cluster, dark ridges are the borders between two of them.
    # the mean, not the sum, so cells on the edge with fewer neighbours do not look paler
    distances = som.distance_map(scaling="mean")

    fig = px.imshow(
        distances,
        color_continuous_scale="Greys",
        labels={"x": "j", "y": "i", "color": "Distance"},
        title="SOM u-matrix",
    )

    fig.update_layout(xaxis_dtick=1, yaxis_dtick=1)

    return fig

def plot_som_case_trajectory(trajectory: pd.DataFrame, state_colors: dict, case_id: str) -> go.Figure:
    # trajectory is what get_som_case_trajectory returns for the case
    return _plot_trajectories(trajectory.assign(case=str(case_id)), state_colors, f"Case {case_id}")

def plot_som_case_trajectories(trajectories: pd.DataFrame, state_colors: dict) -> go.Figure:
    # trajectories is what get_som_case_trajectories returns
    return _plot_trajectories(trajectories, state_colors)

def plot_som_log_trajectory(trajectory: pd.DataFrame, state_colors: dict) -> go.Figure:
    # trajectory is what get_som_log_trajectory returns
    return _plot_trajectories(trajectory.assign(case="log"), state_colors, "Trajectory of the log")


# k-means

def get_kmeans_state_colors(kmeans: KMeans) -> dict:
    # a color for every cluster, clusters with close centers get close colors, see
    # _colors_from_distances
    distances = get_kmeans_state_distances(kmeans)

    return dict(zip(range(kmeans.n_clusters), _colors_from_distances(distances.to_numpy())))

def plot_kmeans_state_frequencies(frequencies: pd.Series, state_colors: dict) -> go.Figure:
    # frequencies is what get_kmeans_state_frequencies returns
    return _plot_cluster_frequencies(frequencies, state_colors, "k-means cluster")

def plot_kmeans_state_distances(distances: pd.DataFrame) -> go.Figure:
    # distances is what get_kmeans_state_distances returns
    return _plot_state_distances(distances, "Distances between k-means clusters")

def plot_kmeans_case_trajectory(trajectory: pd.DataFrame, state_colors: dict, case_id: str) -> go.Figure:
    # trajectory is what get_kmeans_case_trajectory returns for the case
    return _plot_trajectories(trajectory.assign(case=str(case_id)), state_colors, f"Case {case_id}")

def plot_kmeans_case_trajectories(trajectories: pd.DataFrame, state_colors: dict) -> go.Figure:
    # trajectories is what get_kmeans_case_trajectories returns
    return _plot_trajectories(trajectories, state_colors)

def plot_kmeans_log_trajectory(trajectory: pd.DataFrame, state_colors: dict) -> go.Figure:
    # trajectory is what get_kmeans_log_trajectory returns
    return _plot_trajectories(trajectory.assign(case="log"), state_colors, "Trajectory of the log")


# DBSCAN

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

def get_dbscan_state_colors(dbscan: DBSCAN) -> dict:
    # a color for every cluster, clusters whose core rows lie close get close colors, see
    # _colors_from_distances. the noise gets a light grey
    distances = get_dbscan_state_distances(dbscan)
    clusters = range(dbscan.labels_.max() + 1)

    state_colors = dict(zip(clusters, _colors_from_distances(distances.to_numpy())))
    state_colors[-1] = NOISE_COLOR

    return state_colors

def plot_dbscan_state_frequencies(frequencies: pd.Series, state_colors: dict) -> go.Figure:
    # frequencies is what get_dbscan_state_frequencies returns
    return _plot_cluster_frequencies(frequencies, state_colors, "DBSCAN cluster")

def plot_dbscan_state_distances(distances: pd.DataFrame) -> go.Figure:
    # distances is what get_dbscan_state_distances returns
    return _plot_state_distances(distances, "Distances between DBSCAN clusters")

def plot_dbscan_case_trajectory(trajectory: pd.DataFrame, state_colors: dict, case_id: str) -> go.Figure:
    # trajectory is what get_dbscan_case_trajectory returns for the case
    return _plot_trajectories(trajectory.assign(case=str(case_id)), state_colors, f"Case {case_id}")

def plot_dbscan_case_trajectories(trajectories: pd.DataFrame, state_colors: dict) -> go.Figure:
    # trajectories is what get_dbscan_case_trajectories returns
    return _plot_trajectories(trajectories, state_colors)

def plot_dbscan_log_trajectory(trajectory: pd.DataFrame, state_colors: dict) -> go.Figure:
    # trajectory is what get_dbscan_log_trajectory returns
    return _plot_trajectories(trajectory.assign(case="log"), state_colors, "Trajectory of the log")


# drift

def plot_state_distributions(distributions: pd.DataFrame, state_colors: dict) -> go.Figure:
    # distributions is what compute_state_distributions returns, one bar of shares per signal window
    fig = px.area(
        distributions,
        color_discrete_map=_state_names(state_colors),
        labels={"signal_window": "Signal window", "value": "Share", "state": "State"},
        title="State distribution per signal window",
    )

    # plotly draws the dates as their wall time, so the lines get the same
    _window_lines(fig, distributions.index.strftime("%Y-%m-%d %H:%M:%S"))

    return fig

def plot_drift_signal(drift_signal: pd.DataFrame, divergence: str = "kl", reference: str = "previous", threshold: float = None) -> go.Figure:
    # drift_signal is what compute_drift_signal returns: over time, the start of every signal
    # window, the drift score of the signal window, its divergence from the reference
    fig = px.line(
        drift_signal,
        x="signal_window",
        y="drift_score",
        markers=True,
        labels={"signal_window": "Signal window", "drift_score": f"Drift score ({DIVERGENCES[divergence]})"},
        title=f"Drift signal: {DIVERGENCES[divergence]} of every signal window against the {REFERENCES[reference]}",
    )

    # a dashed line at the threshold, the signal windows above it count as drifts
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.3g}",
                      annotation_position="top left")

    return fig

def plot_signal_window_rows(signal_window_rows: pd.DataFrame) -> go.Figure:
    # signal_window_rows is what compute_signal_window_rows returns: over time, the start of every
    # signal window and the number of its rows, events or feature windows
    fig = px.line(
        signal_window_rows,
        x="signal_window",
        y="rows",
        markers=True,
        labels={"signal_window": "Signal window", "rows": "Rows"},
        title="Rows per signal window: the events or feature windows that fall into it",
    )

    return fig


# shared by the functions above

def _colors_from_distances(distances: np.ndarray) -> list:
    # a color for every state such that the color difference follows the distance between the
    # states, so a trajectory only changes its color much when the case or the log moves to a
    # far away state, and a slow drift through states in between shows as a slow change of color.
    # the colors live in oklab, where equal steps look equally different, and
    # 1. classical multidimensional scaling places the states in three dimensions with their
    #    pairwise distances kept as well as possible,
    # 2. a state far from all others is pulled in logarithmically beyond twice the median distance
    #    from the center, so a single outlier does not squeeze all other states into one color,
    # 3. the first axis runs from dark blue to light yellow, which stays readable with color
    #    blindness and in grey, the second from green to magenta and the third is orthogonal to both,
    # 4. one common scale, as large as the srgb colors and a lightness from 0.38 to 0.86 allow,
    #    keeps the ratios of the distances.
    # the colors show how far the states lie from each other relative to the other states, the
    # signs of the axes are fixed, so the same states always get the same colors
    n = len(distances)

    if n == 1:
        return [_oklab_to_hex(np.array([0.62, 0.0, 0.0]))]

    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:3]
    coords = vectors[:, order] * np.sqrt(np.maximum(values[order], 0))
    coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

    # the first state lies on the negative side of every axis
    coords *= np.where(coords[0] > 0, -1.0, 1.0)

    radii = np.linalg.norm(coords, axis=1)
    knee = 2 * np.median(radii)

    if knee > 0:
        coords *= (knee * np.arcsinh(radii / knee) / np.where(radii > 0, radii, 1))[:, None]

    first = np.array([1.0, 0.0, 0.6]) / np.linalg.norm([1.0, 0.0, 0.6])
    second = np.array([0.0, 1.0, 0.0])
    directions = coords @ np.vstack([first, second, np.cross(first, second)])
    center = np.array([0.62, 0.0, 0.0])

    if not np.abs(directions).max() > 0:
        return [_oklab_to_hex(center)] * n

    # the largest common scale that keeps every color displayable, by bisection
    low, high = 0.0, 1.0 / np.abs(directions).max()

    for _ in range(40):
        scale = (low + high) / 2

        if all(_displayable(center + scale * d) for d in directions):
            low = scale
        else:
            high = scale

    return [_oklab_to_hex(center + low * d) for d in directions]

def _oklab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    # oklab to linear srgb (Ottosson, 2020)
    l, m, s = np.array([
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]) @ lab

    return np.array([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]) @ np.array([l ** 3, m ** 3, s ** 3])

def _displayable(lab: np.ndarray) -> bool:
    # inside srgb, and neither too dark to tell apart nor as light as the noise grey
    rgb = _oklab_to_linear_rgb(lab)

    return 0.38 <= lab[0] <= 0.86 and bool(np.all(rgb >= -1e-6) and np.all(rgb <= 1 + 1e-6))

def _oklab_to_hex(lab: np.ndarray) -> str:
    rgb = np.clip(_oklab_to_linear_rgb(lab), 0, 1)
    # the srgb transfer function
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)

    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)

def _state_names(state_colors: dict) -> dict:
    # the state colors keyed by the state names the tables use
    return {
        f"({key[0]}, {key[1]})" if isinstance(key, tuple) else _cluster_name(key): color
        for key, color in state_colors.items()
    }

def _plot_cluster_frequencies(frequencies: pd.Series, state_colors: dict, clusters: str) -> go.Figure:
    # the frequencies are named after what they count, events or windows
    colors = _state_names(state_colors)
    unit = frequencies.name.capitalize()

    # every cluster is drawn, also the ones without events in the range, so the plots of
    # different date ranges can be compared
    frequencies = frequencies.reindex(list(colors), fill_value=0)

    fig = px.bar(
        x=frequencies.index,
        y=frequencies.to_numpy(),
        color=frequencies.index,
        color_discrete_map=colors,
        text_auto=",d" if len(frequencies) <= 10 else False,
        labels={"x": "Cluster", "y": unit},
        title=f"{unit} per {clusters}",
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

def _plot_trajectories(visits: pd.DataFrame, state_colors: dict, title: str = None) -> go.Figure:
    # one row per case, the one that started first on top, and one bar per visit of a
    # state, from when the case entered it until it left
    cases = visits.groupby("case")["start"].min().sort_values().index
    title = title or f"Trajectories of {len(cases):,} cases"

    fig = px.timeline(
        visits,
        x_start="start",
        x_end="end",
        y="case",
        color="state",
        color_discrete_map=_state_names(state_colors),
        category_orders={"case": list(cases)},
        # events of a case's visit, feature windows of the log's
        hover_data=[column for column in ("events", "feature_windows") if column in visits],
        title=title,
    )

    # with hundreds of cases a row is a pixel or two high, borders and gaps would hide
    # the colors, and the plot grows with the number of cases
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False, bargap=0, height=min(max(450, len(cases)), 1000))

    return fig

def _window_lines(fig: go.Figure, windows) -> None:
    # a thin line at the start of every signal window
    fig.update_layout(shapes=[
        dict(type="line", xref="x", yref="paper", x0=window, x1=window, y0=0, y1=1,
             line=dict(color="white", width=0.5))
        for window in windows
    ])
