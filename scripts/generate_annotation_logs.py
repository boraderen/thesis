"""Generate many varied synthetic logs for the annotation dataset.

Writes XES + metadata pairs to data/annotation/training and data/annotation/testing.
Each log is small-to-mid sized and stamped with a randomised drift recipe so the
downstream pipeline sees a diverse mix of perspectives.
"""
from __future__ import annotations

from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rheon


OUTPUT_ROOT = ROOT / "data" / "annotation"
N_TRAIN = 30
N_TEST = 8
BASE_SEED = 1000

DRIFT_SUBTYPES = {
    "control_flow": ["tree_mutation"],
    "data": ["numeric", "categorical"],
    "inter_case": ["arrival_rate", "burstiness", "case_mix", "concurrency"],
    "resource": ["pool_size", "workload_distribution", "reassignment", "service_time", "handover"],
}

REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]


def sample_drifts(rng: random.Random) -> list[dict]:
    """Pick 1-3 sudden drifts across random perspectives."""
    n = rng.randint(1, 3)
    drifts = []
    for _ in range(n):
        perspective = rng.choice(list(DRIFT_SUBTYPES))
        subtype = rng.choice(DRIFT_SUBTYPES[perspective])
        drift = {
            "perspective": perspective,
            "subtype": subtype,
            "drift_type": "sudden",
        }
        if perspective == "control_flow":
            drift["change_proportion"] = round(rng.uniform(0.2, 0.6), 2)
        drifts.append(drift)
    return drifts


def sample_params(rng: random.Random) -> dict:
    """Lightweight randomisation around the demo defaults."""
    return dict(
        num_traces=rng.choice([800, 1200, 1500, 2000]),
        min_trace_length=4,
        max_trace_length=8,
        avg_trace_length=6,
        trace_length_variance=25,
        horizon_min_days=rng.choice([180, 240, 300, 365]),
        horizon_max_days=365,
        min_activities=rng.randint(5, 6),
        max_activities=6,
        tree_depth_min=2,
        tree_depth_max=2,
        sequence_weight=0.85,
        choice_weight=0.10,
        parallel_weight=0.02,
        loop_weight=0.0,
        or_weight=0.01,
        silent_transition_prob=0.0,
        duplicate_activity_prob=0.0,
        num_resources=rng.choice([6, 8, 10]),
        num_roles=rng.choice([2, 3]),
        num_case_types=3,
        regions=REGIONS,
        inter_arrival_mean_min=rng.choice([20, 30, 45]),
        service_time_mean_min=rng.choice([10, 15, 20]),
        service_time_std_min=rng.choice([5, 7, 9]),
        noise_probability=0,
        noise_similar_vs_random=0,
    )


def generate_split(split: str, n_logs: int, seed_offset: int) -> None:
    target_dir = OUTPUT_ROOT / split
    target_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_logs):
        seed = BASE_SEED + seed_offset + i
        rng = random.Random(seed)
        drifts = sample_drifts(rng)
        params = sample_params(rng)
        stem = f"{split}_{i + 1:03d}"
        target_path = target_dir / f"{stem}.xes"
        print(f"[{split}] {stem}: drifts={[(d['perspective'], d['subtype']) for d in drifts]}")
        rheon.generate_log(
            drifts,
            target_path,
            log_name=stem,
            global_seed=seed,
            **params,
        )


if __name__ == "__main__":
    generate_split("training", N_TRAIN, seed_offset=0)
    generate_split("testing", N_TEST, seed_offset=N_TRAIN)
    print("\nDone.")
