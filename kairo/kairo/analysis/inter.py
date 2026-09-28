import pandas as pd

FEATURES = [
    "active_cases",
    "new_arrivals",
    "completions",
    "events_per_case",
    "mean_delta_t",
    "std_delta_t",
    "stalled_cases",
    "attr_means",
    "attr_stds",
    "attr_shares",
]

def compute_features_inter(log: pd.DataFrame, window: str = "1D", features: list = FEATURES, case_attributes: dict = {}, stall_threshold: str = "1D") -> pd.DataFrame:
    # One row per calendar window instead of per event, describing all cases running in it.
    # window is a pandas frequency like "12h", "1D" or "7D". every window from the first to the
    # last event gets a row, also the ones without events, so the rows follow each other in time.
    # case_attributes names attribute columns with their kind, e.g. {"case:amount": "numerical",
    # "case:region": "categorical"}. attr_means and attr_stds cover the numerical ones, attr_shares
    # the categorical ones. stall_threshold is a pandas duration like "12h" or "2D"
    numerical = [c for c, kind in case_attributes.items() if kind == "numerical"]
    categorical = [c for c, kind in case_attributes.items() if kind == "categorical"]

    for column, kind in case_attributes.items():
        if column not in log.columns:
            raise ValueError(f"Case attribute {column} not found in the log, available columns are {list(log.columns)}")

        if kind not in ("numerical", "categorical"):
            raise ValueError(f"Case attribute {column} is {kind}, pick numerical or categorical")

    # numbers can come as text, e.g. from a csv
    log = log.assign(**{column: pd.to_numeric(log[column], errors="coerce") for column in numerical})

    for column in numerical:
        if log[column].isna().all():
            raise ValueError(f"Case attribute {column} holds no numbers, map it as categorical")

    # the gaps look at the previous event of the case, so the events go in chronological
    # order, no matter how the log came in
    log = log.sort_values("time:timestamp", kind="stable")

    # the start of the window every event falls into
    windows = log["time:timestamp"].dt.floor(window)
    starts = pd.date_range(windows.min(), windows.max(), freq=window)

    blocks = []

    if "active_cases" in features:
        blocks.append(compute_active_cases(log, windows))

    if "new_arrivals" in features:
        blocks.append(compute_new_arrivals(log, window))

    if "completions" in features:
        blocks.append(compute_completions(log, window))

    if "events_per_case" in features:
        blocks.append(compute_events_per_case(log, windows))

    if "mean_delta_t" in features:
        blocks.append(compute_mean_delta_t(log, windows))

    if "std_delta_t" in features:
        blocks.append(compute_std_delta_t(log, windows))

    if "stalled_cases" in features:
        blocks.append(compute_stalled_cases(log, starts, window, stall_threshold))

    if "attr_means" in features and numerical:
        blocks.append(compute_attribute_means(log, windows, numerical))

    if "attr_stds" in features and numerical:
        blocks.append(compute_attribute_stds(log, windows, numerical))

    if "attr_shares" in features and categorical:
        blocks.append(compute_attribute_shares(log, windows, categorical))

    if not blocks:
        raise ValueError(
            f"No feature computed, pick from: {FEATURES}. attr_means and attr_stds need "
            "numerical case attributes, attr_shares categorical ones"
        )

    # the start of every window rides along as its timestamp, a window without events gets zeros
    result = pd.concat(blocks, axis=1).reindex(starts).fillna(0.0).astype(float)

    return result.rename_axis("time:timestamp").reset_index()

def compute_active_cases(log: pd.DataFrame, windows: pd.Series) -> pd.Series:
    # Number of distinct cases with at least one event in the window
    return log.groupby(windows)["case:concept:name"].nunique().rename("active_cases")

def compute_new_arrivals(log: pd.DataFrame, window: str) -> pd.Series:
    # Number of cases whose first event falls into the window
    first = log.groupby("case:concept:name")["time:timestamp"].min().dt.floor(window)

    return first.value_counts().rename("new_arrivals")

def compute_completions(log: pd.DataFrame, window: str) -> pd.Series:
    # Number of cases whose last event falls into the window
    last = log.groupby("case:concept:name")["time:timestamp"].max().dt.floor(window)

    return last.value_counts().rename("completions")

def compute_events_per_case(log: pd.DataFrame, windows: pd.Series) -> pd.Series:
    # Events in the window divided by the number of cases active in it
    events = log.groupby(windows).size()
    cases = log.groupby(windows)["case:concept:name"].nunique()

    return (events / cases).rename("events_per_case")

def compute_mean_delta_t(log: pd.DataFrame, windows: pd.Series) -> pd.Series:
    # Mean minutes between an event and the previous event of its case, every gap counts in
    # the window of its later event. the first event of a case has no gap
    gaps = log.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds() / 60

    return gaps.groupby(windows).mean().rename("mean_delta_t")

def compute_std_delta_t(log: pd.DataFrame, windows: pd.Series) -> pd.Series:
    # Standard deviation of the same gaps in minutes
    gaps = log.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds() / 60

    return gaps.groupby(windows).std().rename("std_delta_t")

def compute_stalled_cases(log: pd.DataFrame, starts: pd.DatetimeIndex, window: str, stall_threshold: str) -> pd.Series:
    # Number of cases still running at the end of the window whose most recent event is older
    # than the stall threshold. a case waits between two of its events and its last event
    # completes it, so it is stalled at every window end that lies more than the threshold after
    # one of its events and before the next one. completed cases never count
    threshold = pd.Timedelta(stall_threshold)
    ends = starts + pd.Timedelta(window)

    following = log.groupby("case:concept:name")["time:timestamp"].shift(-1)
    stalling = following - log["time:timestamp"] > threshold

    # a stall begins once the threshold has passed after the event and ends with the next event
    begins = (log.loc[stalling, "time:timestamp"] + threshold).sort_values()
    resumes = following[stalling].sort_values()

    # stalls that began before the window end minus the ones that were over by then
    counts = begins.searchsorted(ends, side="left") - resumes.searchsorted(ends, side="right")

    return pd.Series(counts, index=starts, name="stalled_cases")

def compute_attribute_means(log: pd.DataFrame, windows: pd.Series, attributes: list) -> pd.DataFrame:
    # Mean of every numerical case attribute over the events in the window
    means = log[attributes].groupby(windows).mean()

    return means.add_prefix("attr_means:")

def compute_attribute_stds(log: pd.DataFrame, windows: pd.Series, attributes: list) -> pd.DataFrame:
    # Standard deviation of every numerical case attribute over the events in the window
    stds = log[attributes].groupby(windows).std()

    return stds.add_prefix("attr_stds:")

def compute_attribute_shares(log: pd.DataFrame, windows: pd.Series, attributes: list) -> pd.DataFrame:
    # Share of the window's events carrying every value of a categorical case attribute, one
    # column per value, so the values of an attribute come together as a set
    events = log.groupby(windows).size()

    blocks = []

    for attribute in attributes:
        shares = pd.crosstab(windows, log[attribute]).div(events, axis=0)
        shares.columns = [f"{attribute}={value}" for value in shares.columns]
        blocks.append(shares)

    return pd.concat(blocks, axis=1).add_prefix("attr_shares:")
