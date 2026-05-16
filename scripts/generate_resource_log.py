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
GLOBAL_SEED = 7

# Log size — big and long so drift periods are easy to read.
NUM_TRACES = 6000
MIN_TRACE_LENGTH = 4
MAX_TRACE_LENGTH = 8
AVG_TRACE_LENGTH = 6
TRACE_LENGTH_VARIANCE = 25

# Time horizon (auto-scaled to event count)
HORIZON_MIN_DAYS = 300
HORIZON_MAX_DAYS = 365

# Process complexity — kept simple so the resource signal isn't drowned out
# by control-flow variation.
MIN_ACTIVITIES = 5
MAX_ACTIVITIES = 6
TREE_DEPTH_MIN = 2
TREE_DEPTH_MAX = 2
SEQUENCE_WEIGHT = 0.80
CHOICE_WEIGHT = 0.10
PARALLEL_WEIGHT = 0.01
LOOP_WEIGHT = 0.0
OR_WEIGHT = 0.01
SILENT_TRANSITION_PROB = 0.0
DUPLICATE_ACTIVITY_PROB = 0.0

# Resources
NUM_RESOURCES = 20
NUM_ROLES = 5
NUM_CASE_TYPES = 3
REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]

# Timing model
INTER_ARRIVAL_MEAN_MIN = 30
SERVICE_TIME_MEAN_MIN = 15
SERVICE_TIME_STD_MIN = 7

# Noise
NOISE_PROBABILITY = 0
NOISE_SIMILAR_VS_RANDOM = 0

# Drift configuration — three sudden reassignments stacked with a pool-size
# cut, each at its own change point. Driftify uses 0.8 post-drift dominance,
# so the cumulative effect is very visible on the resource SOM.
DRIFTS = [
    {"subtype": "reassignment", "drift_type": "sudden", "change_proportion": 0.8},
    {"subtype": "reassignment", "drift_type": "sudden", "change_proportion": 0.8},
    {"subtype": "pool_size", "drift_type": "sudden"},
    {"subtype": "reassignment", "drift_type": "sudden", "change_proportion": 0.8},
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
