import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def log() -> pd.DataFrame:
    """A small deterministic log: 40 cases, 4 activities, 3 resources, 2 attrs."""
    rng = np.random.default_rng(7)
    rows = []
    start = pd.Timestamp("2020-01-01", tz="UTC")
    for c in range(40):
        n = int(rng.integers(3, 7))
        t = start + pd.Timedelta(hours=float(rng.uniform(0, 24 * 20)))
        region = ["north", "south"][c % 2]
        for i in range(n):
            rows.append({
                "case:concept:name": f"c{c}",
                "concept:name": f"A{int(rng.integers(0, 4))}",
                "time:timestamp": t,
                "org:resource": f"r{int(rng.integers(0, 3))}",
                "event:duration_min": float(rng.uniform(1, 30)),
                "amount": float(rng.uniform(100, 900)),
                "region": region,
            })
            t += pd.Timedelta(minutes=float(rng.uniform(20, 600)))
    return (
        pd.DataFrame(rows)
        .sort_values(["case:concept:name", "time:timestamp"])
        .reset_index(drop=True)
    )
