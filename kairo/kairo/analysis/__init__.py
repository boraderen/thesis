import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

# carried along with every feature matrix, never part of the maths
META = ["case:concept:name", "time:timestamp"]

DIVERGENCES = {"kl": "KL divergence", "js": "Jensen–Shannon", "tv": "Total variation", "hellinger": "Hellinger"}
REFERENCES = {"previous": "previous window", "recent": "mean of the recent windows", "baseline": "mean of all windows"}


# scaling and PCA

def standardize(feature_matrix: pd.DataFrame, method: str, exclude: list = ["current_act", "past_acts", "act_set"]) -> pd.DataFrame:
    # column groups in exclude are ignored, by default they are set to kairo's standard features but can adjusted for manual usage
    # supported methods are zscore and minmax to 0-1
    # the case id and timestamp come along with every feature matrix, never scaled

    columns = [
        c for c in feature_matrix.columns
        if c not in META and not c.startswith(tuple(exclude))
    ]
    values = feature_matrix[columns]

    if method == "zscore":
        scaled = (values - values.mean()) / values.std()
    elif method == "minmax":
        scaled = (values - values.min()) / (values.max() - values.min())
    else:
        raise ValueError(f"Unknown method: {method}, pick zscore or minmax")

    # a column that never changes has no spread, the division leaves it empty
    scaled = scaled.fillna(0.0)

    result = feature_matrix.copy()
    result[columns] = scaled

    return result

def compute_pca(feature_matrix: pd.DataFrame, num_components: int = None) -> PCA:
    # return PCA object
    # num_components is how many components are fitted, None fits all of them.
    # where to actually cut is decided afterwards, by looking at their variances
    # the meta columns are only there for reference, they never take part in the maths
    pca = PCA(n_components=num_components)
    pca.fit(feature_matrix.drop(columns=META, errors="ignore"))

    return pca

def apply_pca(feature_matrix: pd.DataFrame, pca: PCA, cut_component: int) -> pd.DataFrame:
    # return compressed dataframe based on cut_component
    # the components come out ordered by variance, so cutting keeps the strongest ones
    features = feature_matrix.drop(columns=META, errors="ignore")
    components = pca.transform(features)[:, :cut_component]

    names = [f"pc_{i + 1}" for i in range(components.shape[1])]

    compressed = pd.DataFrame(
        components,
        index=feature_matrix.index,
        columns=names,
    )

    # the meta columns travel on with the components, the way they do with the features
    meta = [c for c in META if c in feature_matrix.columns]

    return pd.concat([feature_matrix[meta], compressed], axis=1)


# SOM

def compute_som(df: pd.DataFrame, size: tuple[int, int] = (5, 5), learning_rate: float = 0.5, distance: str = "euclidean") -> MiniSom:
    # every neuron of the grid becomes one cluster, so a 5x5 map can hold 25 of them.
    # supported distances are euclidean, cosine, manhattan and chebyshev

    # the meta columns are only carried along for reference, the map trains without them
    data = df.drop(columns=META, errors="ignore").to_numpy()

    som = MiniSom(
        x=size[0],
        y=size[1],
        input_len=data.shape[1],
        activation_distance=distance,
        learning_rate=learning_rate
    )

    # start the neurons off on random rows, then show every row once in random order.
    # the fixed seed above keeps the same run giving the same grid positions
    som.random_weights_init(data)
    som.train(data, num_iteration=len(data), random_order=True)

    return som

def get_som_winners(df: pd.DataFrame, som: MiniSom) -> pd.DataFrame:
    # should return df of two columns giving for each row the winning grid coordinates
    data = df.drop(columns=META, errors="ignore").to_numpy()

    winners = pd.DataFrame(
        [som.winner(x) for x in data],
        index=df.index,
        columns=["i", "j"],
    )

    return winners

def get_som_state_frequencies(df: pd.DataFrame, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> pd.Series:
    # how many events every cell holds between the dates, cells without events left out
    counts = _events_between(df, start_date, end_date).groupby(["i", "j"]).size()
    counts.index = [f"({i}, {j})" for i, j in counts.index]

    return counts.rename("events").rename_axis("state")

def get_som_state_distances(som: MiniSom, distance: str = "euclidean") -> pd.DataFrame:
    # distance between the weights of every two neurons, similar states are close
    weights = som.get_weights()
    rows, cols, dims = weights.shape

    names = [f"({i}, {j})" for i in range(rows) for j in range(cols)]
    distances = pairwise_distances(weights.reshape(-1, dims), metric=distance)

    return pd.DataFrame(distances, index=names, columns=names)

def get_som_trajectories(df: pd.DataFrame, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None, max_cases: int = 10000) -> pd.DataFrame:
    # the states every case in the range went through, one row per visit of a state
    events = _events_in_range(df, start_date, end_date, max_cases, ["i", "j"])
    events["state"] = [f"({i}, {j})" for i, j in zip(events["i"], events["j"])]

    return _state_visits(events)

def get_som_case_trajectory(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    return get_som_trajectories(_get_case(df, case_id)).drop(columns="case")


# k-means

def compute_kmeans(df: pd.DataFrame, k: int = 5) -> KMeans:
    # k-means moves every center to the mean of its cluster, which only works with
    # euclidean distance, so there is no distance to choose
    data = df.drop(columns=META, errors="ignore").to_numpy()

    # the fixed seed keeps the same run giving the same cluster numbers
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    return kmeans

def get_kmeans_clusters(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    # the cluster of every row the model was fitted on, so df has to be that same frame
    return pd.DataFrame({"cluster": kmeans.labels_}, index=df.index)

def get_kmeans_state_frequencies(df: pd.DataFrame, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> pd.Series:
    return _cluster_frequencies(df, start_date, end_date)

def get_kmeans_state_distances(kmeans: KMeans) -> pd.DataFrame:
    # distance between every two cluster centers
    names = [_cluster_name(c) for c in range(kmeans.n_clusters)]
    distances = pairwise_distances(kmeans.cluster_centers_)

    return pd.DataFrame(distances, index=names, columns=names)

def get_kmeans_trajectories(df: pd.DataFrame, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None, max_cases: int = 1000) -> pd.DataFrame:
    return _cluster_trajectories(df, start_date, end_date, max_cases)

def get_kmeans_case_trajectory(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    return get_kmeans_trajectories(_get_case(df, case_id)).drop(columns="case")


# DBSCAN

def compute_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5, distance: str = "euclidean") -> DBSCAN:
    # dense regions become clusters: a row with at least min_samples rows (itself
    # included) within eps is a core of one. rows neither core nor within eps of a
    # core are noise and get cluster -1.
    # supported distances are euclidean, cosine, manhattan and chebyshev
    data = df.drop(columns=META, errors="ignore").to_numpy()

    # prefix features repeat a lot, often thousands of rows share one exact vector.
    # dbscan keeps a neighbour list per row, so every copy would list every other copy
    # and memory grows with the square of the copies. fitting on the distinct rows,
    # each weighted by how often it occurs, gives the same clusters
    rows, counts = np.unique(data, axis=0, return_counts=True)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance)
    dbscan.fit(rows, sample_weight=counts)

    if dbscan.labels_.max() < 0:
        raise ValueError(
            "DBSCAN found no clusters, every row is noise. A larger eps or a smaller "
            "min_samples helps, plot_dbscan_k_distance shows a good eps"
        )

    return dbscan

def get_dbscan_clusters(df: pd.DataFrame, dbscan: DBSCAN) -> pd.DataFrame:
    # dbscan cannot place new rows, it only knows the clusters of the rows it was
    # fitted on, so df has to be that same frame. noise rows have cluster -1
    data = df.drop(columns=META, errors="ignore").to_numpy()

    # the model was fitted on the distinct rows, every row takes the cluster of its own
    _, inverse = np.unique(data, axis=0, return_inverse=True)

    return pd.DataFrame({"cluster": dbscan.labels_[inverse]}, index=df.index)

def get_dbscan_state_frequencies(df: pd.DataFrame, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None) -> pd.Series:
    return _cluster_frequencies(df, start_date, end_date)

def get_dbscan_state_distances(dbscan: DBSCAN) -> pd.DataFrame:
    # dbscan has no centers, so every cluster stands in with the mean of its distinct
    # core rows, which the model keeps. noise has no core rows and is left out
    core_clusters = dbscan.labels_[dbscan.core_sample_indices_]
    clusters = range(dbscan.labels_.max() + 1)
    centers = [dbscan.components_[core_clusters == c].mean(axis=0) for c in clusters]

    names = [_cluster_name(c) for c in clusters]
    distances = pairwise_distances(centers, metric=dbscan.metric)

    return pd.DataFrame(distances, index=names, columns=names)

def get_dbscan_trajectories(df: pd.DataFrame, start_date: str | pd.Timestamp = None, end_date: str | pd.Timestamp = None, max_cases: int = 1000) -> pd.DataFrame:
    return _cluster_trajectories(df, start_date, end_date, max_cases)

def get_dbscan_case_trajectory(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    return get_dbscan_trajectories(_get_case(df, case_id)).drop(columns="case")


# drift

def compute_state_distributions(df: pd.DataFrame, window: str = "7D") -> pd.DataFrame:
    # the share of every state among the events of every calendar window, one row per
    # window. window is a pandas frequency like "12h", "1D" or "7D", windows without
    # events are left out
    windows = df["time:timestamp"].dt.floor(window).rename("window")

    if "cluster" in df.columns:
        counts = pd.crosstab(windows, df["cluster"]).rename(columns=_cluster_name)
    else:
        counts = pd.crosstab(windows, [df["i"], df["j"]])
        counts.columns = [f"({i}, {j})" for i, j in counts.columns]

    counts.columns.name = "state"

    return counts.div(counts.sum(axis=1), axis=0)

def compute_divergences(distributions: pd.DataFrame, divergence: str = "kl", reference: str = "previous", lookback: int = 5) -> pd.DataFrame:
    # how far every window's distribution is from a reference: the previous window,
    # the mean of the lookback windows before it, or the mean of all windows. windows
    # without a reference yet get no score
    if reference == "previous":
        references = distributions.shift(1)
    elif reference == "recent":
        references = distributions.rolling(lookback).mean().shift(1)
    elif reference == "baseline":
        references = distributions * 0 + distributions.mean()
    else:
        raise ValueError(f"Unknown reference: {reference}, pick from {list(REFERENCES)}")

    p = distributions.to_numpy()
    q = references.to_numpy()

    if divergence == "kl":
        scores = _kl(p, q)
    elif divergence == "js":
        scores = 0.5 * _kl(p, (p + q) / 2) + 0.5 * _kl(q, (p + q) / 2)
    elif divergence == "tv":
        scores = 0.5 * np.abs(p - q).sum(axis=1)
    elif divergence == "hellinger":
        scores = np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=1))
    else:
        raise ValueError(f"Unknown divergence: {divergence}, pick from {list(DIVERGENCES)}")

    return pd.DataFrame({"window": distributions.index, "score": scores})


# shared by the functions above

def _cluster_name(cluster: int) -> str:
    # dbscan marks its noise rows as cluster -1
    return "noise" if cluster == -1 else str(cluster)

def _get_case(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    # compared as text, so a case id typed in works for number ids too
    case = df[df["case:concept:name"].astype(str) == str(case_id)]

    if case.empty:
        raise ValueError(f"No events found for case {case_id}")

    return case

def _events_between(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    # the events that happened between the dates, either can be left open
    if start_date:
        df = df[df["time:timestamp"] >= start_date]

    if end_date:
        df = df[df["time:timestamp"] <= end_date]

    return df

def _cluster_frequencies(df: pd.DataFrame, start_date, end_date) -> pd.Series:
    counts = _events_between(df, start_date, end_date)["cluster"].value_counts().sort_index()
    counts.index = counts.index.map(_cluster_name)

    return counts.rename("events").rename_axis("state")

def _events_in_range(df: pd.DataFrame, start_date, end_date, max_cases: int, state_columns: list) -> pd.DataFrame:
    # the events of the cases whose first and last event both lie in the range. with
    # more than max_cases of them, the ones that started first
    df = df[["case:concept:name", "time:timestamp"] + state_columns]
    times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])

    if start_date:
        times = times[times["min"] >= start_date]

    if end_date:
        times = times[times["max"] <= end_date]

    cases = times.sort_values("min").index[:max_cases]

    return df[df["case:concept:name"].isin(cases)]

def _cluster_trajectories(df: pd.DataFrame, start_date, end_date, max_cases: int) -> pd.DataFrame:
    events = _events_in_range(df, start_date, end_date, max_cases, ["cluster"])
    events["state"] = events["cluster"].map(_cluster_name)

    return _state_visits(events)

def _state_visits(events: pd.DataFrame) -> pd.DataFrame:
    # consecutive events of a case in the same state are one visit of that state. a
    # visit lasts until the case's next visit starts, the last one until its last event
    events = events.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    case = events["case:concept:name"]
    visit = ((events["state"] != events["state"].shift()) | (case != case.shift())).cumsum()

    visits = events.groupby(visit).agg(
        case=("case:concept:name", "first"),
        state=("state", "first"),
        start=("time:timestamp", "first"),
        events=("state", "size"),
    ).reset_index(drop=True)

    last_events = events.groupby("case:concept:name")["time:timestamp"].max()
    visits["end"] = visits.groupby("case")["start"].shift(-1).fillna(visits["case"].map(last_events))
    visits["duration"] = visits["end"] - visits["start"]

    return visits[["case", "state", "start", "end", "duration", "events"]]

def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    # a tiny share for every state, so a state missing on one side does not divide by 0
    p = p + 1e-9
    q = q + 1e-9

    return (p * np.log(p / q)).sum(axis=1)
