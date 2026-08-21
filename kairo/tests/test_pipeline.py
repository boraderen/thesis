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
    fs = kairo.build_features(log, "intra_case", features=cfg.features, history=cfg.history)
    reduced, _ = kairo.reduce(fs.values(), n_components=2)
    model = kairo.cluster_kmeans(reduced, n_clusters=4, seed=cfg.seed)
    assert (model.state_ids == res.states.state_ids).all()


def test_seed_reproducibility(log):
    a = kairo.run_intra_case(log, kairo.IntraConfig(pca_components=2, window_minutes=1440))
    b = kairo.run_intra_case(log, kairo.IntraConfig(pca_components=2, window_minutes=1440))
    assert (a.states.state_ids == b.states.state_ids).all()
    assert np.allclose(a.signal["score"], b.signal["score"])
