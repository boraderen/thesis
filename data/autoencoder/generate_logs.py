"""Generate a batch of varied intra-case drift logs for autoencoder training.

Every log is built from the same fixed activity alphabet: no process tree
(base or drifted) ever uses more than MAX_ACTIVITIES activities, so rheon's
labels always stay within a..t. This is what lets one autoencoder with a
fixed input size be trained on all of these logs and reused on new ones.

Each log gets 0, 1 or 2 control_flow drifts (the intra-case perspective),
with randomized drift points, modes, tree sizes and operator weights.
Alongside every `intra_XX.xes` rheon writes its `intra_XX_meta.md` ground
truth; `manifest.csv` summarizes the whole batch in one table.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import rheon

OUT_DIR = Path(__file__).parent
NUM_LOGS = 60
MIN_ACTIVITIES = 5
MAX_ACTIVITIES = 20  # keeps every activity label inside a..t
SEED = 42

# operator-weight presets for the process trees, from sequence-heavy to loopy
TREE_WEIGHTS = [
    {"sequence": 0.60, "choice": 0.25, "parallel": 0.10, "loop": 0.05},
    {"sequence": 0.75, "choice": 0.15, "parallel": 0.05, "loop": 0.05},
    {"sequence": 0.45, "choice": 0.40, "parallel": 0.10, "loop": 0.05},
    {"sequence": 0.50, "choice": 0.20, "parallel": 0.25, "loop": 0.05},
    {"sequence": 0.55, "choice": 0.25, "parallel": 0.10, "loop": 0.10},
]


def sample_num_activities(rng: np.random.Generator) -> int:
    return int(rng.integers(MIN_ACTIVITIES, MAX_ACTIVITIES + 1))


def sample_drift(rng: np.random.Generator, start: float, end: float) -> dict:
    """One control_flow drift whose transition lies inside (start, end)."""
    drift = {
        "type": "control_flow",
        "num_activities": sample_num_activities(rng),
        "tree_weights": TREE_WEIGHTS[rng.integers(len(TREE_WEIGHTS))],
    }
    if rng.random() < 0.5:
        drift["mode"] = "sudden"
        drift["drift_point"] = round(float(rng.uniform(start, end)), 3)
    else:
        width = float(rng.uniform(0.05, 0.15))
        left = float(rng.uniform(start, end - width))
        drift["mode"] = "gradual"
        drift["start_point"] = round(left, 3)
        drift["end_point"] = round(left + width, 3)
    return drift


def sample_drifts(rng: np.random.Generator) -> list[dict]:
    n_drifts = rng.choice([0, 1, 2], p=[0.15, 0.55, 0.30])
    if n_drifts == 0:
        return []
    if n_drifts == 1:
        return [sample_drift(rng, 0.25, 0.75)]
    # two drifts in disjoint halves of the horizon so their windows never overlap
    return [sample_drift(rng, 0.15, 0.45), sample_drift(rng, 0.50, 0.85)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    manifest_rows = []

    for i in range(NUM_LOGS):
        name = f"intra_{i:02d}"
        num_traces = int(rng.integers(500, 1201))
        num_activities = sample_num_activities(rng)
        tree_weights = TREE_WEIGHTS[rng.integers(len(TREE_WEIGHTS))]
        drifts = sample_drifts(rng)
        seed = int(rng.integers(0, 2**31 - 1))

        rheon.generate_log(
            drifts,
            str(OUT_DIR / f"{name}.xes"),
            num_traces=num_traces,
            num_activities=num_activities,
            tree_weights=tree_weights,
            seed=seed,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )

        manifest_rows.append({
            "log": name,
            "num_traces": num_traces,
            "num_activities": num_activities,
            "n_drifts": len(drifts),
            "drifts": json.dumps(drifts),
            "seed": seed,
        })
        print(f"{name}: {num_traces} traces, {num_activities} activities, {len(drifts)} drift(s)")

    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote {NUM_LOGS} logs and manifest.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
