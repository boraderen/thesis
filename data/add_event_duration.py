import pandas as pd
import pm4py

INPUT_PATH = "./data/input.xes"
OUTPUT_PATH = "./data/output.xes"
DURATION_COLUMN = "event:duration_min"

log = pm4py.read_xes(INPUT_PATH)

# the duration of every event in minutes, from its start to its completion
start = pd.to_datetime(log["start_timestamp"], utc=True)
end = pd.to_datetime(log["time:timestamp"], utc=True)
log[DURATION_COLUMN] = (end - start).dt.total_seconds() / 60

pm4py.write_xes(log, OUTPUT_PATH)
