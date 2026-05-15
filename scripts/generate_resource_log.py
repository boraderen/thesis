from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.driftify.config import GeneratorConfig
from scripts.driftify.script_api import run_generation


# Output
OUTPUT_PATH = "data/resource/"
NUM_LOGS = 1
GLOBAL_SEED = 42

# Log size
NUM_TRACES = 1000
MIN_TRACE_LENGTH = 3
MAX_TRACE_LENGTH = 5
AVG_TRACE_LENGTH = 4
TRACE_LENGTH_VARIANCE = 35

# Time horizon (auto-scaled to event count)
HORIZON_MIN_DAYS = 270
HORIZON_MAX_DAYS = 365

# Process complexity
MIN_ACTIVITIES = 4
MAX_ACTIVITIES = 7
TREE_DEPTH_MIN = 2
TREE_DEPTH_MAX = 3
SEQUENCE_WEIGHT = 0.70
CHOICE_WEIGHT = 0.10
PARALLEL_WEIGHT = 0.01
LOOP_WEIGHT = 0.01
OR_WEIGHT = 0.01
SILENT_TRANSITION_PROB = 0.05
DUPLICATE_ACTIVITY_PROB = 0.0

# Resources
NUM_RESOURCES = 20
NUM_ROLES = 5
NUM_CASE_TYPES = 3
REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]

# Timing model
INTER_ARRIVAL_MEAN_MIN = 5
SERVICE_TIME_MEAN_MIN = 10
SERVICE_TIME_STD_MIN = 5

# Noise
NOISE_PROBABILITY = 0
NOISE_SIMILAR_VS_RANDOM = 0

# Drift configuration
DRIFTS = [
    {"subtype": "reassignment", "drift_type": "sudden", "change_proportion": 0.3},
    #{"subtype": "pool_size", "drift_type": "gradual_linear"},
    #{"subtype": "handover", "drift_type": "incremental"},
    #{"subtype": "workload_distribution", "drift_type": "recurring"},
]


def build_config() -> GeneratorConfig:
    return GeneratorConfig.from_uppercase(globals())


def generate_logs():
    return run_generation(
        build_config(),
        DRIFTS,
        filename_prefix="resource",
        default_perspective="resource",
    )


if __name__ == "__main__":
    generate_logs()
