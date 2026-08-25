import numpy as np

import kairo


def test_run_intra_all_methods(log):
    for method in ("som", "kmeans", "dbscan"):
        cfg = kairo.IntraConfig(clustering=method, window_minutes=1440, pca_components=3)
        res = kairo.run_intra_case(log, cfg)
        assert len(res.trajectories) == len(log)
        assert res.states.n_states >= 1
        assert (res.signal["score"] >= 0).all()


def test_run_resource_and_inter(log):
    rres = kairo.run_resource(log, kairo.ResourceConfig(window_minutes=1440, clustering="kmeans", n_clusters=3))
    assert len(rres.signal) == len(rres.features.matrix)
    ires = kairo.run_inter_case(log, kairo.InterConfig(window_minutes=1440, grid=(2, 2),
                                                       numeric_attrs=("amount",)))
    assert ires.states.n_states == 4
    assert len(ires.states.dominant) == 4


def test_pipeline_matches_steps(log):
    cfg = kairo.IntraConfig(clustering="kmeans", n_clusters=4, pca_components=2, window_minutes=1440)
    res = kairo.run_intra_case(log, cfg)
    fs = kairo.features.build_features(log, "intra_case", features=cfg.features, history=cfg.history)
    reduced, _ = kairo.analysis.reduce(fs.values(), n_components=2)
    model = kairo.analysis.cluster_kmeans(reduced, n_clusters=4, seed=cfg.seed)
    assert (model.state_ids == res.states.state_ids).all()


def test_seed_reproducibility(log):
    a = kairo.run_intra_case(log, kairo.IntraConfig(pca_components=2, window_minutes=1440))
    b = kairo.run_intra_case(log, kairo.IntraConfig(pca_components=2, window_minutes=1440))
    assert (a.states.state_ids == b.states.state_ids).all()
    assert np.allclose(a.signal["score"], b.signal["score"])


def test_save_figure_handles_time_axes(log, tmp_path):
    """Time-axis figures carry pandas Timestamps the image writer cannot serialize."""
    res = kairo.run_intra_case(log, kairo.IntraConfig(window_minutes=1440, pca_components=2))
    fig = kairo.plot_drift_signal(res.signal, title="drift")
    kairo.add_window_boundaries(fig, res.signal["window_start"])
    out = kairo.save_figure(fig, tmp_path / "signal.png")
    assert out.exists() and out.stat().st_size > 3000


def test_window_vector_shift_references(log):
    """previous localises a jump, baseline spreads it, recent sits between."""
    import numpy as np
    import pandas as pd

    mat = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
    ws = pd.date_range("2020-01-01", periods=5, freq="D")

    prev = kairo.analysis.window_vector_shift(ws, mat, reference="previous")["score"].to_numpy()
    assert prev[3] > 0 and prev[4] == 0 and prev[0] == 0

    recent = kairo.analysis.window_vector_shift(ws, mat, reference="recent", lookback=2)["score"].to_numpy()
    assert recent[3] > 0 and recent[4] > 0  # window 4's reference still straddles the jump

    base = kairo.analysis.window_vector_shift(ws, mat, reference="baseline")["score"].to_numpy()
    assert (base > 0).all()  # every window differs from the overall mean

    # the default is unchanged behaviour
    assert np.allclose(kairo.analysis.window_vector_shift(ws, mat)["score"], prev)


def test_windowed_configs_carry_the_reference(log):
    cfg = kairo.ResourceConfig(window_minutes=1440, clustering="kmeans", n_clusters=3,
                               signal_reference="baseline")
    res = kairo.run_resource(log, cfg)
    assert res.config.signal_reference == "baseline"
    assert (res.signal["score"] > 0).any()
