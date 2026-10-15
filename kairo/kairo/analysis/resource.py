import pandas as pd

# every feature group with its label and what it computes, the single source for the docs,
# the dashboard and the language models. a handover is an event whose previous event in the
# same case another resource executed
GLOSSARY = {
    "res_events": ("Events per resource",
                   "One column per resource: the number of events the resource executed in the feature window."),
    "res_cases": ("Active cases per resource",
                  "One column per resource: the number of distinct cases with at least one event of the "
                  "resource in the feature window."),
    "res_durations": ("Mean event duration per resource",
                      "One column per resource: the mean event duration in minutes over the resource's events "
                      "in the feature window, 0 without events. Needs an event duration column."),
    "res_waits": ("Mean wait into resource",
                  "One column per resource: the mean minutes between an event and the previous event of its "
                  "case, over the handovers to the resource in the feature window, 0 without handovers. A "
                  "handover is an event whose previous event in the same case another resource executed."),
    "act_res_shares": ("Activity-resource event shares",
                       "One column per pair a by r that occurs in the log: the events of activity a that "
                       "resource r executed in the feature window, divided by all events of a in the feature "
                       "window, 0 if a has none."),
    "handover_shares": ("Handover shares",
                        "One column per resource pair r1 -> r2 that occurs in the log: the handovers from r1 to "
                        "r2 in the feature window, divided by all handovers from r1 in the feature window, 0 if "
                        "r1 has none. A handover counts in the feature window of the receiving event."),
}

FEATURES = list(GLOSSARY)

def compute_features_resource(log: pd.DataFrame, feature_window: str = "1D", features: list = FEATURES, resources: list = None) -> pd.DataFrame:
    # One row per feature window instead of per event, describing how the resources worked in it.
    # feature_window is a pandas frequency like "12h", "1D" or "7D". every feature window from the first to the
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

    # the start of the feature window every event falls into
    windows = log["time:timestamp"].dt.floor(feature_window)
    starts = pd.date_range(windows.min(), windows.max(), freq=feature_window)

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

    # the start of every feature window rides along as its timestamp, a feature window without events
    # (or without handovers etc.) gets zeros
    result = pd.concat(blocks, axis=1).reindex(starts).fillna(0.0).astype(float)

    return result.rename_axis("time:timestamp").reset_index()

def compute_events_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Number of events every resource executed in the feature window
    counts = pd.crosstab(windows, log["org:resource"]).reindex(columns=resources, fill_value=0)

    return counts.add_prefix("res_events:")

def compute_cases_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Number of distinct cases every resource touched in the feature window
    cases = log.groupby([windows, log["org:resource"]])["case:concept:name"].nunique().unstack(fill_value=0)
    cases = cases.reindex(columns=resources, fill_value=0)

    return cases.add_prefix("res_cases:")

def compute_durations_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Mean duration of the events every resource executed in the feature window, the event duration
    # column holds minutes. a resource without events in the feature window gets 0
    durations = pd.to_numeric(log["event_duration"], errors="coerce")

    means = durations.groupby([windows, log["org:resource"]]).mean().unstack()
    means = means.reindex(columns=resources).fillna(0.0)

    return means.add_prefix("res_durations:")

def compute_waits_per_resource(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Mean minutes between an event and the previous event of its case, over the events every
    # resource executed in the feature window. only events handed over from another resource count
    previous = log.groupby("case:concept:name")[["org:resource", "time:timestamp"]].shift(1)
    handed_over = previous["org:resource"].notna() & (previous["org:resource"] != log["org:resource"])

    waits = (log["time:timestamp"] - previous["time:timestamp"]).dt.total_seconds() / 60
    waits = waits[handed_over]

    means = waits.groupby([windows[handed_over], log.loc[handed_over, "org:resource"]]).mean().unstack()
    means = means.reindex(columns=resources).fillna(0.0)

    return means.add_prefix("res_waits:")

def compute_activity_resource_shares(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Share of every activity's events in the feature window that a resource executed, so the shares of
    # an activity add up to 1 over all resources. pairs that never occur get no column
    counts = pd.crosstab(windows, [log["concept:name"], log["org:resource"]])

    # all events of the pair's activity in the feature window, whoever executed them
    totals = counts.T.groupby(level=0).transform("sum").T
    shares = (counts / totals).fillna(0.0)

    shares = shares.loc[:, shares.columns.get_level_values(1).isin(resources)]
    shares.columns = [f"{activity} by {resource}" for activity, resource in shares.columns]

    return shares.add_prefix("act_res_shares:")

def compute_handover_shares(log: pd.DataFrame, windows: pd.Series, resources: list) -> pd.DataFrame:
    # Share of a resource's handovers in the feature window that go to another resource, so the shares
    # of a resource add up to 1 over all receivers. a handover is a case moving from one resource
    # to another, it counts in the feature window of the receiving event. pairs that never occur get no column
    giver = log.groupby("case:concept:name")["org:resource"].shift(1).rename("giver")
    receiver = log["org:resource"].rename("receiver")
    handed_over = giver.notna() & (giver != receiver)

    counts = pd.crosstab(windows[handed_over], [giver[handed_over], receiver[handed_over]])

    # all handovers of the pair's giver in the feature window, to whoever
    totals = counts.T.groupby(level=0).transform("sum").T
    shares = (counts / totals).fillna(0.0)

    picked = shares.columns.get_level_values(0).isin(resources) & shares.columns.get_level_values(1).isin(resources)
    shares = shares.loc[:, picked]
    shares.columns = [f"{r1} -> {r2}" for r1, r2 in shares.columns]

    return shares.add_prefix("handover_shares:")
