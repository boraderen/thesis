"""Reproducible kairo pipeline runs and analysis helpers for the kairo-drift-analyst skill.

Import it in a script or notebook:

    import sys; sys.path.insert(0, "skills/kairo-drift-analyst/scripts")
    import kairo_pipeline as kp
    result = kp.run_pipeline(kp.load_config("my_config.json"))

or run it from the repository root as a command line tool:

    uv run python skills/kairo-drift-analyst/scripts/kairo_pipeline.py my_config.json --out analyses/run1

A config describes one pipeline of one perspective as a plain dict, see
assets/config_template.json. Everything here only calls kairo's public functions (plus pandas,
numpy and scikit-learn for the sweeps), so every run can be repeated by hand with the same values.
No function calls a language model.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

import kairo

META = ["case:concept:name", "time:timestamp"]
PERSPECTIVES = ("intra", "resource", "inter")
METHODS = ("som", "kmeans", "dbscan")
BINARY_GROUPS = ["current_act", "past_acts", "act_set"]
ANALYSIS_FOLDERS = {
    "padella": "padella",
    "rheon_single_drift": "rheon_single",
    "rheon_multi_drift": "rheon_multi",
    "rheon_mixed_drift": "rheon_mixed",
}


# configs and environment

def load_config(path: str | Path) -> dict:
    """A config from a JSON file."""
    return json.loads(Path(path).read_text())


def environment() -> dict:
    """The versions and the git commit a later run needs to reproduce the results."""
    def package(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    status = git("status", "--porcelain")

    return {
        "kairo": getattr(kairo, "__version__", None),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": package("scikit-learn"),
        "minisom": package("minisom"),
        "pm4py": package("pm4py"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_uncommitted_changes": None if status is None else bool(status),
    }


# log

def load_log(path: str | Path, event_duration_from_timestamps: bool = False, **columns) -> pd.DataFrame:
    """kairo.read_log plus the timestamp conversion a CSV needs. columns are the role arguments
    of read_log, e.g. resource="org:resource", event_duration="event:duration_min". A log with
    start and completion timestamps but no duration column, e.g. a Padella log, gets its
    event_duration in minutes from the two timestamps with event_duration_from_timestamps."""
    log = kairo.read_log(str(path), **columns)

    for column in ("time:timestamp", "start_timestamp"):
        if column in log.columns and not pd.api.types.is_datetime64_any_dtype(log[column]):
            log[column] = pd.to_datetime(log[column], utc=True, format="mixed")

    if event_duration_from_timestamps:
        if "start_timestamp" not in log.columns:
            raise ValueError("event_duration_from_timestamps needs start_timestamp mapped in read_log")
        log["event_duration"] = (log["time:timestamp"] - log["start_timestamp"]).dt.total_seconds() / 60

    return log


def log_overview(log: pd.DataFrame, freq: str = "1D") -> dict:
    """The numbers to start from: kairo's log statistics, the events and new cases per calendar
    window of freq, empty windows, and how the log starts and ends. A long tail without new cases
    or a long warm-up before the first case completes makes the first and last signal windows
    unreliable."""
    stats = kairo.compute_log_stats(log)
    times = log["time:timestamp"]
    starts = log.groupby("case:concept:name")["time:timestamp"].min()
    ends = log.groupby("case:concept:name")["time:timestamp"].max()

    windows = pd.date_range(times.min().floor(freq), times.max().floor(freq), freq=freq)
    events = times.dt.floor(freq).value_counts().reindex(windows, fill_value=0)
    arrivals = starts.dt.floor(freq).value_counts().reindex(windows, fill_value=0)

    span_days = (times.max() - times.min()).total_seconds() / 86400

    return {
        "stats": {key: value for key, value in stats.items() if key not in ("activity_counts", "resource_counts")},
        "activity_counts": stats["activity_counts"].to_dict(),
        "resource_counts": stats["resource_counts"].head(20).to_dict(),
        "columns": list(log.columns),
        "span_days": span_days,
        "cases_per_day": stats["cases"] / span_days if span_days else None,
        "events_per_day": stats["events"] / span_days if span_days else None,
        f"events_per_{freq}": events.describe().to_dict(),
        f"new_cases_per_{freq}": arrivals.describe().to_dict(),
        f"empty_{freq}_windows": int((events == 0).sum()),
        "first_event": times.min(),
        "first_completion": ends.min(),
        "last_arrival": starts.max(),
        "last_event": times.max(),
        "tail_without_new_cases_days": (times.max() - starts.max()).total_seconds() / 86400,
        "warm_up_before_first_completion_days": (ends.min() - times.min()).total_seconds() / 86400,
    }


# pipeline steps

def compute_features(log: pd.DataFrame, perspective: str, spec: dict | None = None) -> pd.DataFrame:
    """The kairo features of a perspective. spec holds groups and the feature parameters, missing
    values take kairo's defaults. Without groups, all groups are computed, for resource without
    res_durations if the log has no event duration column."""
    spec = spec or {}
    groups = spec.get("groups")

    if perspective == "intra":
        return kairo.compute_features_intra(
            log,
            features=groups or list(kairo.INTRA_GLOSSARY),
            sliding_window_size=spec.get("sliding_window_size", 0),
        )

    if perspective == "resource":
        if not groups:
            groups = [g for g in kairo.RESOURCE_GLOSSARY if g != "res_durations" or "event_duration" in log.columns]

        return kairo.compute_features_resource(
            log,
            feature_window=spec.get("feature_window", "1D"),
            features=groups,
            resources=spec.get("resources"),
        )

    if perspective == "inter":
        return kairo.compute_features_inter(
            log,
            feature_window=spec.get("feature_window", "1D"),
            features=groups or list(kairo.INTER_GLOSSARY),
            case_attributes=spec.get("case_attributes", {}),
            stall_threshold=spec.get("stall_threshold", "1D"),
        )

    raise ValueError(f"Unknown perspective {perspective}, pick from {PERSPECTIVES}")


def suggest_pca_cut(pca) -> int:
    """A heuristic cut: the knee of the cumulative explained variance, i.e. the number of
    components farthest above the straight line from no component to all of them. Check the
    loadings before trusting it."""
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(ratios)])

    if len(cumulative) < 3 or cumulative[-1] <= 0:
        return 1

    x = np.arange(len(cumulative)) / (len(cumulative) - 1)
    y = cumulative / cumulative[-1]

    return max(1, int(np.argmax(y - x)))


def reduce(features: pd.DataFrame, scaling: dict | None = None, pca: dict | None = None) -> dict:
    """Scaling and PCA, both optional. pca may give "cut": an integer, or "elbow"/None for
    suggest_pca_cut. Returns the scaled frame, the clustering input and the PCA results."""
    scaling = scaling or {}
    method = scaling.get("method")

    if method:
        scaled = kairo.standardize(features, method=method, exclude=scaling.get("exclude", BINARY_GROUPS))
    else:
        scaled = features

    if not pca:
        return {"scaled": scaled, "reduced": scaled, "pca": None, "errors": None, "cut": None}

    model = kairo.compute_pca(scaled, num_components=pca.get("num_components"))
    errors = kairo.compute_pca_reconstruction_errors(scaled, model)
    cut = pca.get("cut")
    cut = suggest_pca_cut(model) if cut in (None, "elbow") else int(cut)

    return {
        "scaled": scaled,
        "reduced": kairo.apply_pca(scaled, model, cut_component=cut),
        "pca": model,
        "errors": errors,
        "cut": cut,
    }


def cluster(reduced: pd.DataFrame, spec: dict) -> dict:
    """States with SOM, k-means or DBSCAN, plus their colors, distances, frequencies and
    scores. spec: {"method": "som", "size": [r, c], "learning_rate": 0.5, "distance": ...},
    {"method": "kmeans", "k": 5} or {"method": "dbscan", "eps": 0.5, "min_samples": 5, "distance": ...}."""
    method = spec["method"]
    distance = spec.get("distance", "euclidean")
    size = None

    if method == "som":
        size = tuple(spec.get("size", (5, 5)))
        model = kairo.compute_som(reduced, size=size, learning_rate=spec.get("learning_rate", 0.5), distance=distance)
        labels = kairo.get_som_states(reduced, model)
        colors = kairo.get_som_state_colors(model, distance)
        distances = kairo.get_som_state_distances(model, distance)
    elif method == "kmeans":
        distance = "euclidean"
        model = kairo.compute_kmeans(reduced, k=spec.get("k", 5))
        labels = kairo.get_kmeans_states(reduced, model)
        colors = kairo.get_kmeans_state_colors(model)
        distances = kairo.get_kmeans_state_distances(model)
    elif method == "dbscan":
        model = kairo.compute_dbscan(reduced, eps=spec.get("eps", 0.5), min_samples=spec.get("min_samples", 5), distance=distance)
        labels = kairo.get_dbscan_states(reduced, model)
        colors = kairo.get_dbscan_state_colors(model)
        distances = kairo.get_dbscan_state_distances(model)
    else:
        raise ValueError(f"Unknown method {method}, pick from {METHODS}")

    states = pd.concat([reduced, labels], axis=1)

    return {
        "method": method,
        "model": model,
        "labels": labels,
        "states": states,
        "colors": colors,
        "distances": distances,
        "frequencies": getattr(kairo, f"get_{method}_state_frequencies")(states),
        "scores": score_states(reduced, labels, method, model, size, distance),
    }


def score_states(reduced: pd.DataFrame, labels: pd.DataFrame, method: str, model=None, size=None, distance: str = "euclidean") -> dict:
    """The clustering scores that fit the method, and how many states are used."""
    if "cluster" in labels.columns:
        clusters = labels["cluster"]
        scores = {"states": int(clusters[clusters != -1].nunique())}

        if method == "dbscan":
            scores["noise_share"] = float((clusters == -1).mean())
    else:
        scores = {"states": int(len(labels.drop_duplicates()))}

    scores["silhouette"] = kairo.compute_silhouette(reduced, labels, distance=distance)
    scores["calinski_harabasz"] = kairo.compute_calinski_harabasz(reduced, labels)

    if method == "som":
        scores["quantization_error"] = kairo.compute_som_quantization_error(reduced, labels, model, distance)
        scores["occupancy_entropy"] = kairo.compute_som_occupancy_entropy(labels, size)

    if method == "dbscan":
        scores["dbcv"] = kairo.compute_dbcv(reduced, labels, distance=distance)

    return scores


def drift_signals(states: pd.DataFrame, specs: list, log: pd.DataFrame | None = None) -> list:
    """One drift signal per spec {"signal_window", "divergence", "reference", "lookback"}, with
    the state distributions and the signal window report."""
    signals = []

    for spec in specs:
        spec = {
            "signal_window": spec["signal_window"],
            "divergence": spec.get("divergence", "kl"),
            "reference": spec.get("reference", "previous"),
            "lookback": spec.get("lookback", 5),
        }
        distributions = kairo.compute_state_distributions(states, signal_window=spec["signal_window"])
        signal = kairo.compute_drift_signal(
            distributions, divergence=spec["divergence"], reference=spec["reference"], lookback=spec["lookback"]
        )
        signals.append({
            "spec": spec,
            "distributions": distributions,
            "signal": signal,
            "report": signal_window_report(states, signal, spec["signal_window"], log),
        })

    return signals


def run_pipeline(config: dict, log: pd.DataFrame | None = None, features: pd.DataFrame | None = None) -> dict:
    """The whole pipeline of one perspective. Without log, the log of config["log"] is read.
    With features, the feature step is skipped (custom features, declare them in the entry).
    The returned config holds the resolved PCA cut."""
    config = copy.deepcopy(config)
    perspective = config["perspective"]

    if log is None and (features is None or config.get("log", {}).get("path")):
        log_spec = config["log"]
        log = load_log(log_spec["path"], log_spec.get("event_duration_from_timestamps", False),
                       **log_spec.get("read_log", {}))

    if features is None:
        features = compute_features(log, perspective, config.get("features"))

    reduction = reduce(features, config.get("scaling"), config.get("pca"))

    if reduction["cut"] is not None:
        config["pca"]["cut"] = reduction["cut"]

    clustering = cluster(reduction["reduced"], config["clustering"])
    signals = drift_signals(clustering["states"], config.get("drift", []), log)

    return {
        "config": config,
        "environment": environment(),
        "log": log,
        "features": features,
        **reduction,
        "clustering": clustering,
        "signals": signals,
    }


# checks of the signal windows

def signal_window_report(states: pd.DataFrame, signal: pd.DataFrame, signal_window: str, log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per signal window: drift score, rows (events or feature windows), the share of the window
    covered by the log, new cases (with log) and flags for windows to distrust: partial (first or
    last, truncated), thin (few rows) and no new cases."""
    times = states["time:timestamp"]
    length = pd.Timedelta(signal_window)
    begin, end = times.min(), times.max()

    if "case:concept:name" not in states.columns:
        # a feature window row stands for the whole feature window after its start
        end = end + times.sort_values().diff().min()

    report = signal.set_index("signal_window")
    report["rows"] = times.dt.floor(signal_window).value_counts().reindex(report.index, fill_value=0)

    # the part of every signal window between the first and the last row of the log
    window_starts = pd.Series(report.index, index=report.index)
    window_ends = window_starts + length
    covered = window_ends.where(window_ends <= end, end) - window_starts.where(window_starts >= begin, begin)
    report["coverage"] = (covered / length).clip(lower=0, upper=1)

    if log is not None:
        new_cases = log.groupby("case:concept:name")["time:timestamp"].min().dt.floor(signal_window).value_counts()
        report["new_cases"] = new_cases.reindex(report.index, fill_value=0)

    median_rows = report["rows"].median()
    flags = []

    for position, (start, row) in enumerate(report.iterrows()):
        flag = []

        if pd.isna(row["drift_score"]):
            flag.append("no reference")
        if row["coverage"] < 0.99:
            flag.append("partial window")
        if row["rows"] < 0.25 * median_rows:
            flag.append("thin")
        if "new_cases" in report and row["new_cases"] == 0:
            flag.append("no new cases")
        if position == 0:
            flag.append("first")
        if position == len(report) - 1:
            flag.append("last")

        flags.append(", ".join(flag))

    report["flags"] = flags

    return report


def top_windows(report: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The signal windows with the highest drift scores, with their checks."""
    return report.dropna(subset=["drift_score"]).nlargest(n, "drift_score")


# parameter search

def sweep_kmeans(reduced: pd.DataFrame, ks) -> pd.DataFrame:
    """Silhouette, Calinski-Harabasz and the share of the smallest state for every k."""
    rows = []

    for k in ks:
        labels = kairo.get_kmeans_states(reduced, kairo.compute_kmeans(reduced, k=k))
        rows.append({
            "k": k,
            "silhouette": kairo.compute_silhouette(reduced, labels),
            "calinski_harabasz": kairo.compute_calinski_harabasz(reduced, labels),
            "smallest_state_share": float(labels["cluster"].value_counts(normalize=True).min()),
        })

    return pd.DataFrame(rows).set_index("k")


def sweep_som(reduced: pd.DataFrame, sizes, learning_rate: float = 0.5, distance: str = "euclidean") -> pd.DataFrame:
    """Used cells, quantization error, occupancy entropy, silhouette and Calinski-Harabasz for
    every grid size."""
    rows = []

    for size in sizes:
        size = tuple(size)
        som = kairo.compute_som(reduced, size=size, learning_rate=learning_rate, distance=distance)
        labels = kairo.get_som_states(reduced, som)
        rows.append({
            "size": f"{size[0]}x{size[1]}",
            "used_cells": int(len(labels.drop_duplicates())),
            "quantization_error": kairo.compute_som_quantization_error(reduced, labels, som, distance),
            "occupancy_entropy": kairo.compute_som_occupancy_entropy(labels, size),
            "silhouette": kairo.compute_silhouette(reduced, labels, distance=distance),
            "calinski_harabasz": kairo.compute_calinski_harabasz(reduced, labels),
        })

    return pd.DataFrame(rows).set_index("size")


def sweep_dbscan(reduced: pd.DataFrame, eps_values, min_samples: int, distance: str = "euclidean") -> pd.DataFrame:
    """Clusters, noise share, silhouette and DBCV for every eps."""
    rows = []

    for eps in eps_values:
        try:
            labels = kairo.get_dbscan_states(reduced, kairo.compute_dbscan(reduced, eps=eps, min_samples=min_samples, distance=distance))
        except ValueError:
            rows.append({"eps": eps, "clusters": 0, "noise_share": 1.0, "silhouette": np.nan, "dbcv": np.nan})
            continue

        rows.append({
            "eps": eps,
            "clusters": int(labels.loc[labels["cluster"] != -1, "cluster"].nunique()),
            "noise_share": float((labels["cluster"] == -1).mean()),
            "silhouette": kairo.compute_silhouette(reduced, labels, distance=distance),
            "dbcv": kairo.compute_dbcv(reduced, labels, distance=distance),
        })

    return pd.DataFrame(rows).set_index("eps")


def k_distance_summary(reduced: pd.DataFrame, min_samples: int, distance: str = "euclidean") -> dict:
    """The numbers behind plot_dbscan_k_distance: quantiles of the distance of every row to its
    min_samples-th nearest row (itself counted, as in DBSCAN) and a heuristic knee, computed
    below the 99th percentile so that a few far outliers do not hide it."""
    from sklearn.neighbors import NearestNeighbors

    data = reduced.drop(columns=META, errors="ignore").to_numpy()
    distances = NearestNeighbors(n_neighbors=min_samples, metric=distance).fit(data).kneighbors(data)[0][:, -1]
    distances = np.sort(distances)

    trimmed = distances[distances <= np.quantile(distances, 0.99)]
    knee = None

    if len(trimmed) >= 3 and trimmed[-1] > trimmed[0]:
        x = np.linspace(0, 1, len(trimmed))
        y = (trimmed - trimmed[0]) / (trimmed[-1] - trimmed[0])
        knee = float(trimmed[int(np.argmax(x - y))])

    return {
        "min_samples": min_samples,
        "quantiles": {q: float(np.quantile(distances, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "max": float(distances[-1]),
        "knee_eps": knee,
    }


def suggest_k_from_som(som, labels: pd.DataFrame, ks) -> pd.DataFrame:
    """Groups of SOM cells as candidates for k: k-means on the prototypes of the occupied cells,
    weighted by their rows, with the silhouette of the prototypes for every k. Read it together
    with plot_som_u_matrix, whose dark ridges separate the groups."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    weights = som.get_weights()
    occupied = labels.groupby(["i", "j"]).size()
    prototypes = np.array([weights[i, j] for i, j in occupied.index])
    rows = []

    for k in ks:
        if not 2 <= k < len(prototypes):
            continue

        model = KMeans(n_clusters=k, random_state=0).fit(prototypes, sample_weight=occupied.to_numpy())
        groups = {}

        for (i, j), group in zip(occupied.index, model.labels_):
            groups.setdefault(int(group), []).append(f"({i}, {j})")

        # identical prototypes can leave fewer groups than k, then there is no silhouette
        found = len(set(model.labels_))
        silhouette = float(silhouette_score(prototypes, model.labels_)) if 2 <= found < len(prototypes) else np.nan

        rows.append({"k": k, "silhouette": silhouette, "groups": groups})

    return pd.DataFrame(rows).set_index("k") if rows else pd.DataFrame(columns=["silhouette", "groups"])


# explaining states and shifts

def state_names(labels: pd.DataFrame) -> pd.Series:
    """The state of every row as kairo names it: "(i, j)", "0", "1", ... or "noise"."""
    if "cluster" in labels.columns:
        return labels["cluster"].map(lambda c: "noise" if c == -1 else str(c))

    return "(" + labels["i"].astype(str) + ", " + labels["j"].astype(str) + ")"


def state_profiles(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The mean of every (unscaled) feature per state: what a state stands for. features and
    labels must share their index, which they do for the frames of one run."""
    values = features.drop(columns=META, errors="ignore")

    return values.groupby(state_names(labels).reindex(values.index)).mean().T


def feature_shift(features: pd.DataFrame, split, before: str | None = None, after: str | None = None, top: int = 10) -> pd.DataFrame:
    """Which features change most at a date: mean before and after split, the difference in
    standard deviations of the whole log, largest first. before and after are optional pandas
    durations that limit the ranges, e.g. "30D"."""
    times = features["time:timestamp"]
    split = _timestamp(split, times)
    values = features.drop(columns=META, errors="ignore")

    lower = split - pd.Timedelta(before) if before else times.min()
    upper = split + pd.Timedelta(after) if after else times.max() + pd.Timedelta("1ns")

    left = values[(times >= lower) & (times < split)]
    right = values[(times >= split) & (times < upper)]

    shift = pd.DataFrame({"before": left.mean(), "after": right.mean(), "std": values.std()})
    shift["shift_in_std"] = (shift["after"] - shift["before"]) / shift["std"].replace(0, np.nan)

    return shift.reindex(shift["shift_in_std"].abs().sort_values(ascending=False).index).head(top)


def _timestamp(value, reference: pd.Series) -> pd.Timestamp:
    # a date given as text gets the time zone of the log
    timestamp = pd.Timestamp(value)
    zone = getattr(reference.dt, "tz", None)

    if zone is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(zone)

    return timestamp


# text and figures

def abstractions(result: dict) -> dict:
    """kairo's text abstractions of every step of a run, to read the results as text."""
    clustering = result["clustering"]
    method = clustering["method"]
    texts = {}

    if result["log"] is not None:
        texts["log"] = kairo.abstract_log_stats(kairo.compute_log_stats(result["log"]))

    texts["features"] = kairo.abstract_features(result["features"])

    if result["pca"] is not None:
        texts["pca"] = kairo.abstract_pca(result["pca"], result["cut"])

    scores = [{"type": name, "value": value} for name, value in clustering["scores"].items()]
    texts["states"] = kairo.abstract_states(clustering["frequencies"], clustering["distances"], clustering["colors"], scores)

    if "case:concept:name" not in clustering["states"].columns:
        trajectory = getattr(kairo, f"get_{method}_log_trajectory")(clustering["states"])
        texts["log_trajectory"] = kairo.abstract_log_trajectory(trajectory)

    for i, signal in enumerate(result["signals"]):
        spec = signal["spec"]
        texts[f"state_distributions_{i}"] = kairo.abstract_state_distributions(signal["distributions"])
        texts[f"drift_signal_{i}"] = kairo.abstract_drift_signal(signal["signal"], spec["divergence"], spec["reference"], spec["lookback"])

    return texts


def figures(result: dict) -> dict:
    """kairo's plots of a run, as plotly figures by name."""
    clustering = result["clustering"]
    method = clustering["method"]
    states = clustering["states"]
    figs = {}

    if result["pca"] is not None:
        figs["pca_variances"] = kairo.plot_pca_variances(result["pca"], result["cut"])
        figs["pca_reconstruction_errors"] = kairo.plot_pca_reconstruction_errors(result["errors"], result["cut"])

    if method == "som":
        size = tuple(result["config"]["clustering"].get("size", (5, 5)))
        figs["som_heatmap"] = kairo.plot_som_heatmap(states, size)
        figs["som_u_matrix"] = kairo.plot_som_u_matrix(clustering["model"])
        figs["som_state_colors"] = kairo.plot_som_state_colors(states, clustering["colors"])
    else:
        figs["state_frequencies"] = getattr(kairo, f"plot_{method}_state_frequencies")(clustering["frequencies"], clustering["colors"])
        figs["state_distances"] = getattr(kairo, f"plot_{method}_state_distances")(clustering["distances"])

    if "case:concept:name" not in states.columns:
        trajectory = getattr(kairo, f"get_{method}_log_trajectory")(states)
        figs["log_trajectory"] = getattr(kairo, f"plot_{method}_log_trajectory")(trajectory, clustering["colors"])

    for i, signal in enumerate(result["signals"]):
        spec = signal["spec"]
        figs[f"state_distributions_{i}"] = kairo.plot_state_distributions(signal["distributions"], clustering["colors"])
        figs[f"drift_signal_{i}"] = kairo.plot_drift_signal(signal["signal"], spec["divergence"], spec["reference"])

    return figs


def save_figures(figs: dict, folder: str | Path) -> list:
    """Every figure as PNG (needs kaleido) to look at, else as HTML."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []

    for name, fig in figs.items():
        try:
            path = folder / f"{name}.png"
            fig.write_image(path, width=1200, height=600)
        except Exception:
            path = folder / f"{name}.html"
            fig.write_html(path)

        paths.append(str(path))

    return paths


def save_drift_signal_plot(signal: dict, path: str | Path, threshold: float | None = None) -> str:
    """One drift signal of result["signals"] as PNG for the analysis entry, with the threshold
    as a dashed line."""
    spec = signal["spec"]
    fig = kairo.plot_drift_signal(signal["signal"], spec["divergence"], spec["reference"])

    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path, width=900, height=300)

    return str(path)


def analysis_folder(log_path: str | Path, root: str | Path = "analyses") -> Path:
    """The folder of a log in the analyses: <root>/<benchmark>/<log name>, where the benchmark
    follows from the folder of the log (padella, rheon_single_drift, rheon_multi_drift,
    rheon_mixed_drift, any other folder is other) and the log name is the log file name without
    its extension. It holds one numbered result folder per analysis, 1, 2, 3, ..."""
    log_path = Path(log_path)

    return Path(root) / ANALYSIS_FOLDERS.get(log_path.parent.name, "other") / log_path.stem


def new_result_folder(log_path: str | Path, root: str | Path = "analyses") -> Path:
    """Creates and returns the result folder of a new analysis of the log: the folder of the log
    (see analysis_folder) with the next number after the highest one in it, 1 for the first
    analysis. It holds results.md, plots/ and drift_signals/. The plain mkdir claims the number,
    so agents that analyze the same log in parallel never get the same folder."""
    base = analysis_folder(log_path, root)
    base.mkdir(parents=True, exist_ok=True)
    k = max((int(path.name) for path in base.iterdir() if path.is_dir() and path.name.isdigit()), default=0) + 1

    while True:
        try:
            (base / str(k)).mkdir()
            return base / str(k)
        except FileExistsError:
            k += 1


def signal_files(folder: str | Path, perspective: str) -> dict:
    """The paths of the plot and the CSV of one drift signal in a result folder (see
    new_result_folder). Both share the name <log name>_<perspective>: the log without its
    extension, which is the name of the folder above the result folder, and the perspective
    (intra, resource or inter)."""
    if perspective not in PERSPECTIVES:
        raise ValueError(f"Unknown perspective: {perspective}, pick from {list(PERSPECTIVES)}")

    folder = Path(folder)
    stem = f"{folder.parent.name}_{perspective}"

    return {
        "plot": folder / "plots" / f"{stem}.png",
        "csv": folder / "drift_signals" / f"{stem}.csv",
    }


def save_drift_signal_csv(signal: dict, path: str | Path) -> str:
    """The signal window report of one drift signal of result["signals"] as CSV: signal_window,
    drift_score, rows, coverage and flags, the first signal window without a score."""
    report = signal["report"].reset_index()[["signal_window", "drift_score", "rows", "coverage", "flags"]]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)

    return str(path)


def save_drift_signal(signal: dict, folder: str | Path, perspective: str, threshold: float | None = None) -> dict:
    """The plot and the CSV of one drift signal under their shared name in a result folder (see
    signal_files)."""
    files = signal_files(folder, perspective)
    save_drift_signal_plot(signal, files["plot"], threshold)
    save_drift_signal_csv(signal, files["csv"])

    return {key: str(path) for key, path in files.items()}


# summary

def summary(result: dict, top: int = 5) -> dict:
    """The essentials of a run as plain data: config, environment, sizes, PCA,
    scores, state frequencies and, per drift signal, its highest windows with their checks."""
    clustering = result["clustering"]
    pca = result["pca"]

    signals = []

    for signal in result["signals"]:
        scores = signal["signal"]["drift_score"]
        signals.append({
            "spec": signal["spec"],
            "signal_windows": int(len(signal["signal"])),
            "median_score": _number(scores.median()),
            "top_windows": _records(top_windows(signal["report"], top)),
        })

    return _plain({
        "config": result["config"],
        "environment": result["environment"],
        "rows": int(len(result["features"])),
        "feature_columns": int(len(result["features"].columns.difference(META))),
        "pca": None if pca is None else {
            "cut": result["cut"],
            "explained_variance_kept": float(pca.explained_variance_ratio_[: result["cut"]].sum()),
        },
        "scores": clustering["scores"],
        "state_frequencies": clustering["frequencies"].to_dict(),
        "signals": signals,
    })


def _records(frame: pd.DataFrame) -> list:
    return [{"signal_window": str(index), **row} for index, row in frame.to_dict(orient="index").items()]


def _number(value):
    return None if pd.isna(value) else float(value)


def _plain(value):
    # json cannot write numpy numbers, timestamps or tuples as keys
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    return value


# command line

def main() -> None:
    parser = argparse.ArgumentParser(description="Run one kairo pipeline from a JSON config.")
    parser.add_argument("config", help="path to the JSON config")
    parser.add_argument("--out", required=True, help="folder for summary, abstractions, signals and figures")
    parser.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = parser.parse_args()

    result = run_pipeline(load_config(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(json.dumps(result["config"], indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(summary(result), indent=2, default=str))
    (out / "abstractions.txt").write_text("\n\n".join(f"## {name}\n{text}" for name, text in abstractions(result).items()))

    if result["log"] is not None:
        (out / "log_overview.json").write_text(json.dumps(_plain(log_overview(result["log"])), indent=2, default=str))

    for i, signal in enumerate(result["signals"]):
        signal["report"].to_csv(out / f"signal_{i}.csv")
        signal["distributions"].to_csv(out / f"distributions_{i}.csv")

    if not args.no_figures:
        save_figures(figures(result), out / "figures")

    print(f"run -> {out}")


if __name__ == "__main__":
    main()
