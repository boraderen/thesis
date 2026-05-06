from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.driftify.config import GeneratorConfig, make_rng


@pytest.fixture
def small_config() -> GeneratorConfig:
    return GeneratorConfig(
        num_logs=1,
        num_traces=24,
        min_trace_length=3,
        max_trace_length=8,
        avg_trace_length=5,
        min_activities=4,
        max_activities=8,
        num_resources=6,
        num_roles=3,
        num_case_types=2,
        horizon_min_days=2,
        horizon_max_days=3,
        noise_probability=0.0,
        global_seed=123,
    )


@pytest.fixture
def rng():
    return make_rng(123)
