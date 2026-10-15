import pandas as pd

# every feature group with its label and what it computes, the single source for the docs,
# the dashboard and the language models. the prefix of an event is its case up to and
# including the event
GLOSSARY = {
    "act_freqs": ("Activity frequencies",
                  "One column per activity: how often the activity occurs in the prefix, divided by the "
                  "number of events in the prefix."),
    "df_counts": ("Directly-follows counts",
                  "One column per directly-follows pair a -> b that occurs in the log: how often b "
                  "directly follows a in the prefix."),
    "act_set": ("Activity set",
                "One column per activity: 1 if the activity occurs in the prefix, else 0."),
    "case_progress": ("Case progress",
                      "One column: the position of the event in its case, divided by the number of events "
                      "of the whole case, so the last event has 1."),
    "current_act": ("Current activity",
                    "One column per activity: 1 for the activity of the event, else 0."),
    "past_acts": ("Past activities",
                  "One block of activity columns per step back k = 1, ..., h, where h is the sliding window "
                  "size: 1 for the activity of the event k steps earlier in the same case, else 0. All 0 if "
                  "the case has fewer than k earlier events."),
}

FEATURES = list(GLOSSARY)

def compute_features_intra(log: pd.DataFrame, features: list = FEATURES, sliding_window_size: int = 0) -> pd.DataFrame:
    # Every feature looks at the events of a case up to and including the current one,
    # so the features are computed in chronological order, no matter how the log came in
    order = log.index
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

    # the case id and the timestamp ride along with the features, the index says
    # which event of the log a row belongs to
    meta = log[["case:concept:name", "time:timestamp"]]

    # concat leaves the blocks lying next to each other in memory, the copy packs
    # them into one array, which makes standardizing and PCA a lot faster later on.
    # reindex hands the rows back in the order they came in, not the sorted one
    result = pd.concat([meta] + blocks, axis=1).reindex(order).copy()

    return result

def compute_activity_freqs(log: pd.DataFrame) -> pd.DataFrame:
    # Share of every activity among the events of the case so far
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    counts = one_hot.groupby(log["case:concept:name"]).cumsum()
    seen_events = log.groupby("case:concept:name").cumcount() + 1

    freqs = counts.div(seen_events, axis=0)

    return freqs.add_prefix("act_freqs:")

def compute_directly_follow_counts(log: pd.DataFrame) -> pd.DataFrame:
    # How often every directly follows pair happened in the case so far.
    # The first event of a case has no predecessor, its pair stays empty
    previous = log.groupby("case:concept:name")["concept:name"].shift(1)
    pairs = previous + " -> " + log["concept:name"]

    one_hot = pd.get_dummies(pairs, dtype=float)
    counts = one_hot.groupby(log["case:concept:name"]).cumsum()

    return counts.add_prefix("df_counts:")

def compute_activity_set(log: pd.DataFrame) -> pd.DataFrame:
    # 1 for every activity the case has already executed, ignoring how often
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    seen = one_hot.groupby(log["case:concept:name"]).cummax()

    return seen.add_prefix("act_set:")

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
    # Shifting inside the case keeps the sliding window from reaching into another case
    one_hot = pd.get_dummies(log["concept:name"], dtype=float)

    blocks = []

    for lag in range(1, n + 1):
        previous = one_hot.groupby(log["case:concept:name"]).shift(lag).fillna(0.0)
        blocks.append(previous.add_prefix(f"past_acts_{lag}:"))

    if not blocks:
        return pd.DataFrame(index=log.index)

    return pd.concat(blocks, axis=1)
