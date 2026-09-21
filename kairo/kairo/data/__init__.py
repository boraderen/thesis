import pm4py
import pandas as pd
from pathlib import Path

def read_log(
    path: str,
    case_id: str = "case:concept:name",
    activity: str =  "concept:name",
    timestamp: str = "time:timestamp",
    event_id: str = None,
    start_timestamp: str = None,
    resource: str = None,
    event_duration: str = None
) -> pd.DataFrame:
    
    path_str = path
    path_obj = Path(path)
    
    if path_obj.suffix.lower() == ".xes":
        df = pm4py.read_xes(path)
    elif path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Rename the standart columns
    df = df.rename(columns={
        case_id: "case:concept:name",
        activity:  "concept:name",
        timestamp: "time:timestamp",
    })

    if event_id != None:
        df = df.rename(columns={event_id: "event_id"})

    if start_timestamp != None:
        df = df.rename(columns={start_timestamp: "start_timestamp"})

    if resource != None:
        df = df.rename(columns={resource: "org:resource"})

    if event_duration != None:
        df = df.rename(columns={event_duration: "event_duration"})
            
    return df


def compute_log_stats(
    log: pd.DataFrame
) -> dict:
    
    cases = log["case:concept:name"].nunique()
    events = len(log)
    activities = log["concept:name"].nunique()
    
    resources = log["org:resource"].nunique()
    start = log["time:timestamp"].min()
    end = log["time:timestamp"].max()
    span_minutes = (end - start).total_seconds() / 60
    
    case_times = log.groupby("case:concept:name")["time:timestamp"].agg(
        ["min", "max"]
    )
    tpt_days = (
        case_times["max"] - case_times["min"]
    ).dt.total_seconds() / 86400

    case_lengths = log.groupby("case:concept:name").size()

    activity_counts = log["concept:name"].value_counts()
    resource_counts = log["org:resource"].value_counts()

    stats = {
        "cases": cases,
        "events": events,
        "activities": activities,
        "resources": resources,

        "start": start,
        "end": end,
        "span_minutes": float(span_minutes),

        "tpt_days_min": float(tpt_days.min()),
        "tpt_days_mean": float(tpt_days.mean()),
        "tpt_days_median": float(tpt_days.median()),
        "tpt_days_max": float(tpt_days.max()),

        "length_min": int(case_lengths.min()),
        "length_mean": float(case_lengths.mean()),
        "length_median": float(case_lengths.median()),
        "length_max": int(case_lengths.max()),

        "activity_counts": activity_counts,
        "resource_counts": resource_counts,
    }

    return stats