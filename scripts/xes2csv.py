import pm4py
import pandas as pd

XES_PATH = "data/resource/resource_001.xes"
CSV_PATH = "data/resource/resource_001.csv"

log = pm4py.read_xes(XES_PATH)

log["time:timestamp"] = pd.to_datetime(log["time:timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
log["start_timestamp"] = pd.to_datetime(log["start_timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

log.to_csv(CSV_PATH, index=False)

print(f"Done. {len(log)} events written to {CSV_PATH}")
print(f"Sample timestamp: {log['time:timestamp'].iloc[0]}")
print(f"Sample start_timestamp: {log['start_timestamp'].iloc[0]}")