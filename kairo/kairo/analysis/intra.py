import pandas as pd

FEATURES = [
    "act_freqs",
    "df_counts",
    "act_set",
    "case_counts",
    "case_progress",
    "current_act",
    "past_acts",
]

def compute_features_intra(log: pd.DataFrame, features: list = [""], sliding_window_size: int = 0) -> pd.DataFrame:
    # Every feature looks at the events of a case up to and including the current one,
    # so the log has to be in chronological order first
    log = log.sort_values("time:timestamp", kind="stable")

    blocks = []

    if "act_freqs" in features:
        blocks.append(compute_activity_freqs(log))

    if "df_counts" in features:
        blocks.append(compute_directly_follow_counts(log))

    if "act_set" in features:
        blocks.append(compute_activity_set(log))

    if "case_progress" in features:
        blocks.append(compute_case_progress(log))

    if "current_act" in features:
        blocks.append(compute_current_activity(log))

    if "past_acts" in features:
        blocks.append(compute_past_activities(log, n=sliding_window_size))

    if not blocks:
        raise ValueError(f"No known feature requested, pick from: {FEATURES}")

    result = pd.concat(blocks, axis=1)

    return result


def compute_activity_freqs(log: pd.DataFrame) -> pd.DataFrame:
    # Share of every activity among the events of the case so far
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    counts = one_hot.groupby(log["case:concept:name"]).cumsum()
    seen_events = log.groupby("case:concept:name").cumcount() + 1

    freqs = counts.div(seen_events, axis=0)

    return freqs.add_prefix("act_freq:")

def compute_directly_follow_counts(log: pd.DataFrame) -> pd.DataFrame:
    # How often every directly follows pair happened in the case so far.
    # The first event of a case has no predecessor, its pair stays empty
    previous = log.groupby("case:concept:name")["concept:name"].shift(1)
    pairs = previous + " -> " + log["concept:name"]

    one_hot = pd.get_dummies(pairs, dtype=float)
    counts = one_hot.groupby(log["case:concept:name"]).cumsum()

    return counts.add_prefix("df_count:")

def compute_activity_set(log: pd.DataFrame) -> pd.DataFrame:
    # 1 for every activity the case has already executed, ignoring how often
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    seen = one_hot.groupby(log["case:concept:name"]).cummax()

    return seen.add_prefix("act_seen:")

def compute_case_progress(log: pd.DataFrame) -> pd.DataFrame:
    # Position of the event inside its case, as a fraction of the whole case
    position = log.groupby("case:concept:name").cumcount() + 1
    case_length = log.groupby("case:concept:name")["concept:name"].transform("size")

    progress = position / case_length

    return progress.to_frame("case_progress")

def compute_current_activity(log: pd.DataFrame) -> pd.DataFrame:
    # One column per activity, 1 for the activity of the event itself
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    return one_hot.add_prefix("current_act:")

def compute_past_activities(log: pd.DataFrame, n: int) -> pd.DataFrame:
    # n is sliding window size: one block of activity columns per previous event.
    # Shifting inside the case keeps the window from reaching into another case
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    blocks = []

    for lag in range(1, n + 1):
        previous = one_hot.groupby(log["case:concept:name"]).shift(lag).fillna(0.0)
        blocks.append(previous.add_prefix(f"past_act_{lag}:"))

    if not blocks:
        return pd.DataFrame(index=log.index)

    return pd.concat(blocks, axis=1)
