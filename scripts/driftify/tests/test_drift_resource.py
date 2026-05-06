from __future__ import annotations

import pytest

from scripts.driftify.config import RESOURCE_KEY, ROLE_KEY, sample_horizon
from scripts.driftify.drift.common import sample_drift_plans
from scripts.driftify.drift.resource import build_resource_runtime


@pytest.mark.parametrize(
    "subtype,expected_key",
    [
        ("reassignment", "new_dominant_resource"),
        ("pool_size", "new_pool_size"),
        ("handover", "new_dominant_target"),
        ("workload_distribution", "heavy_resources"),
    ],
)
def test_resource_drift_subtypes_change_state(small_config, rng, subtype, expected_key):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"subtype": subtype, "drift_type": "sudden", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
        default_perspective="resource",
    )
    runtime = build_resource_runtime(small_config, plans, small_config.activity_pool[:5], rng)

    assert len(runtime.drifts) == 1
    assert expected_key in runtime.drifts[0].details
    assert runtime.metadata[0].perspective == "resource"
    assert runtime.metadata[0].affected_columns == [RESOURCE_KEY, ROLE_KEY]


def test_resource_gradual_runtime_samples_both_versions(small_config, rng):
    start, end = sample_horizon(small_config, rng)
    plans = sample_drift_plans(
        [{"subtype": "pool_size", "drift_type": "gradual_linear", "change_point": 0.5}],
        config=small_config,
        horizon_start=start,
        horizon_end=end,
        rng=rng,
        default_perspective="resource",
    )
    runtime = build_resource_runtime(small_config, plans, small_config.activity_pool[:5], rng)
    plan = plans[0]

    old_size = len(runtime.drifts[0].versions[0].active_resources)
    new_size = len(runtime.drifts[0].versions[1].active_resources)
    sampled_sizes = {
        len(runtime.state_for(plan.change_point, rng).active_resources)
        for _ in range(100)
    }

    assert {old_size, new_size}.issubset(sampled_sizes)
