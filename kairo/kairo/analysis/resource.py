import pandas as pd

FEATURES = [
    "res_events",
    "res_cases",
    "res_durations",
    "res_waits",
    "act_res_shares",
    "handover_shares",
]

def compute_features_resource(log: pd.DataFrame, window: str = "1D", features: list = FEATURES, resources: list = None) -> pd.DataFrame:
    # One row per calendar window instead of per event, describing how the resources worked in it.
    # window is a pandas frequency like "12h", "1D" or "7D". every window from the first to the
    # last event gets a row, also the ones without events, so the rows follow each other in time.
    # resources picks the columns, None keeps all. every resource still counts in the shares and
    # waits of the picked ones
    if "org:resource" not in log.columns:
        raise ValueError("The resource features need a resource column, map one when reading the log")

    if "res_durations" in features and "event_duration" not in log.columns:
        raise ValueError("res_durations needs an event duration column, map one when reading the log or leave the feature out")

    all_resources = sorted(log["org:resource"].dropna().unique())
    resources = resources or all_resources
    missing = [r for r in resources if r not in all_resources]

    if missing:
        raise ValueError(f"Resources {missing} not found in the log")

    # waits and handovers look at the previous event of the case, so the events go in
    # chronological order, no matter how the log came in
    log = log.sort_values("time:timestamp", kind="stable")

    # the start of the window every event falls into
    windows = log["time:timestamp"].dt.floor(window)
    starts = pd.date_range(windows.min(), windows.max(), freq=window)

    blocks = []

    if "res_events" in features:
        blocks.append(compute_events_per_resource(log, windows, resources))

    if "res_cases" in features:
        blocks.append(compute_cases_per_resource(log, windows, resources))

    if "res_durations" in features:
        blocks.append(compute_durations_per_resource(log, windows, resources))

    if "res_waits" in features:
        blocks.append(compute_waits_per_resource(log, windows, resources))

    if "act_res_shares" in features:
        blocks.append(compute_activity_resource_shares(log, windows, resources))

    if "handover_shares" in features:
        blocks.append(compute_handover_shares(log, windows, resources))

    if not blocks:
        raise ValueError(f"No known feature requested, pick from: {FEATURES}")

    # the start of every window rides along as its timestamp, a window without events
    # (or without handovers etc.) gets zeros
    result = pd.concat(blocks, axis=1).reindex(starts).fillna(0.0).astype(float)

    return result.rename_axis("time:timestamp").reset_index()

def compute_events_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Number of events every resource executed in the window
    counts = pd.crosstab(windows, log["org:resource"]).reindex(columns=resources, fill_value=0)

    return counts.add_prefix("res_events:")

def compute_cases_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Number of distinct cases every resource touched in the window
    cases = log.groupby([windows, log["org:resource"]])["case:concept:name"].nunique().unstack(fill_value=0)
    cases = cases.reindex(columns=resources, fill_value=0)

    return cases.add_prefix("res_cases:")

def compute_durations_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Mean duration of the events every resource executed in the window, the event duration
    # column holds minutes. a resource without events in the window gets 0
    durations = pd.to_numeric(log["event_duration"], errors="coerce")

    means = durations.groupby([windows, log["org:resource"]]).mean().unstack()
    means = means.reindex(columns=resources).fillna(0.0)

    return means.add_prefix("res_durations:")

def compute_waits_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Mean minutes between an event and the previous event of its case, over the events every
    # resource executed in the window. only events handed over from another resource count
    previous = log.groupby("case:concept:name")[["org:resource", "time:timestamp"]].shift(1)
    handed_over = previous["org:resource"].notna() & (previous["org:resource"] != log["org:resource"])

    waits = (log["time:timestamp"] - previous["time:timestamp"]).dt.total_seconds() / 60
    waits = waits[handed_over]

    means = waits.groupby([windows[handed_over], log.loc[handed_over, "org:resource"]]).mean().unstack()
    means = means.reindex(columns=resources).fillna(0.0)

    return means.add_prefix("res_waits:")

def compute_activity_resource_shares(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Share of every activity's events in the window that a resource executed, so the shares of
    # an activity add up to 1 over all resources. pairs that never occur get no column
    counts = pd.crosstab(windows, [log["concept:name"], log["org:resource"]])

    # all events of the pair's activity in the window, whoever executed them
    totals = counts.T.groupby(level=0).transform("sum").T
    shares = (counts / totals).fillna(0.0)

    shares = shares.loc[:, shares.columns.get_level_values(1).isin(resources)]
    shares.columns = [f"{activity} by {resource}" for activity, resource in shares.columns]

    return shares.add_prefix("act_res_shares:")

def compute_handover_shares(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Share of a resource's handovers in the window that go to another resource, so the shares
    # of a resource add up to 1 over all receivers. a handover is a case moving from one resource
    # to another, it counts in the window of the receiving event. pairs that never occur get no column
    giver = log.groupby("case:concept:name")["org:resource"].shift(1).rename("giver")
    receiver = log["org:resource"].rename("receiver")
    handed_over = giver.notna() & (giver != receiver)

    counts = pd.crosstab(windows[handed_over], [giver[handed_over], receiver[handed_over]])

    # all handovers of the pair's giver in the window, to whoever
    totals = counts.T.groupby(level=0).transform("sum").T
    shares = (counts / totals).fillna(0.0)

    picked = shares.columns.get_level_values(0).isin(resources) & shares.columns.get_level_values(1).isin(resources)
    shares = shares.loc[:, picked]
    shares.columns = [f"{r1} -> {r2}" for r1, r2 in shares.columns]

    return shares.add_prefix("handover_shares:")
