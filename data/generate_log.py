import rheon
from datetime import datetime

drifts = [
        {"type": "reassignment", "mode": "sudden", "drift_point": 0.2},
        {"type": "duration", "mode": "sudden", "drift_point": 0.4, "resources": ["res_01", "res_02"], "factor": 2.5},
        #{"type": "pool_size", "mode": "sudden", "drift_point": 0.6, "delta": -2},
        {"type": "workload", "mode": "sudden", "drift_point": 0.8, "workload_factor": 1.6},
    ]


rheon.generate_log(
    drifts,
    "./data/example.xes",
    num_traces=2000,
    num_activities=12,
    num_resources=7,
    seed=7,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31)
)
