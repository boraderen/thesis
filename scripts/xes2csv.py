import pm4py
import pandas as pd

file_path = "../logs/event-log.xes"

log = pm4py.read_xes(file_path)
df = pm4py.convert_to_dataframe(log)

# Check what columns you have
print(df.columns.tolist())
# Should include 'case:concept:name', 'concept:name', 'time:timestamp'

# Rename to Celonis-friendly names
df = df.rename(columns={
    'case:concept:name': 'Case_ID',
    'concept:name': 'Activity',
    'time:timestamp': 'Timestamp'
})

file_path = file_path[0:len(file_path)-3]
file_path = file_path + "csv"

df.to_csv(file_path, index=False)