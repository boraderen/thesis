import rheon
from datetime import datetime

drifts = [
        #{"type": "reassignment", "mode": "sudden", "drift_point": 0.5},
        #{"type": "reassignment", "mode": "sudden", "drift_point": 0.33}
        #{"type": "duration", "mode": "sudden", "drift_point": 0.4, "resources": ["res_01", "res_02"], "factor": 2.5},
        #{"type": "pool_size", "mode": "sudden", "drift_point": 0.6, "delta": -2},
        #{"type": "workload", "mode": "sudden", "drift_point": 0.8, "workload_factor": 1.6},
        #{"type": "control_flow", "mode": "sudden", "drift_point": 0.50, "num_activities": 9, "tree_weights": {"sequence": 0.50, "choice": 0.30, "parallel": 0.15, "loop": 0.05}}
        #{"type": "amount", "mode": "sudden", "drift_point": 0.50, "mean": 3000.0, "variance": 40000.0},
        {"type": "waiting_time", "mode": "sudden", "drift_point": 0.3, "mean": 45.0, "variance": 80.0,
    }
    ]


rheon.generate_log(
    drifts,
    "./data/amount.xes",
    num_traces=5000,
    num_activities=7,
    num_resources=7,
    seed=7,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31)
)
