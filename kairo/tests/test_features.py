import numpy as np
import pandas as pd

import kairo


def test_intra_only_requested_features(log):
    fs = kairo.features.build_features(log, "intra_case", features=("progress", "current"))
    assert list(fs.groups) == ["progress", "current"]
    assert fs.matrix.shape[1] == 1 + log["concept:name"].nunique()


def test_intra_prefix_semantics():
    log = pd.DataFrame({
        "case:concept:name": ["c", "c", "c"],
        "concept:name": ["A", "B", "A"],
        "time:timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"], utc=True),
    })
    fs = kairo.features.build_features(log, "intra_case", features=("activity_freq", "bigram", "vocab", "progress"))
    m = fs.frame
    # third event: A seen twice of three, B once; transitions A→B and B→A once of two
    assert np.isclose(m["activity_freq:A"].iloc[2], 2 / 3)
    assert np.isclose(m["bigram:A→B"].iloc[2], 1 / 2)
    assert m["vocab:B"].iloc[2] == 1.0
    assert np.isclose(m["progress_ratio"].iloc[1], 2 / 3)
    assert m["bigram:A→B"].iloc[0] == 0.0  # no transitions yet at the first event


def test_resource_share_denominators_ignore_filter(log):
    full = kairo.features.build_features(log, "resource", features=("ho",), window_minutes=1440)
    part = kairo.features.build_features(log, "resource", features=("ho",), window_minutes=1440,
                                resources=("r0", "r1"))
    shared = [c for c in part.columns if c in full.columns]
    assert shared and np.allclose(part.matrix[shared].to_numpy(), full.matrix[shared].to_numpy())


def test_inter_attribute_features(log):
    fs = kairo.features.build_features(log, "inter_case", window_minutes=1440,
                              numeric_attrs=("amount",), categorical_attrs=("region",))
    assert "attr_mean:amount" in fs.columns
    share_cols = [c for c in fs.columns if c.startswith("attr_share:region=")]
    sums = fs.matrix[share_cols].sum(axis=1)
    assert ((np.isclose(sums, 1.0)) | (np.isclose(sums, 0.0))).all()
