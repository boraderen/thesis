import json

import kairo


def test_abstractions_respect_max_len(log):
    res = kairo.run_intra_case(log, kairo.IntraConfig(window_minutes=1440, pca_components=2))
    for text in (
        kairo.abstract_result(res, max_len=1200),
        kairo.abstract_states(res.states, max_len=300),
        kairo.abstract_drift_signal(res.signal, max_len=300),
        kairo.abstract_log_statistics(kairo.log_statistics(log), max_len=300),
    ):
        assert isinstance(text, str) and 0 < len(text) <= 1200


def test_build_prompt_offline(log):
    res = kairo.run_intra_case(log, kairo.IntraConfig(window_minutes=1440, pca_components=2))
    prompt = kairo.llm.build_prompt("Where is the drift?", result=res, log=log)
    assert "<knowledge>" in prompt and "<analysis>" in prompt and "<question>" in prompt


def test_nlp_to_config_with_fake_executor():
    def fake(prompt, **kwargs):
        return '```json\n{"clustering": "dbscan", "eps": 0.3, "grid": [4, 4]}\n```'

    cfg = kairo.llm.nlp_to_config("tight dbscan", "intra_case", executor=fake)
    assert cfg.clustering == "dbscan" and cfg.eps == 0.3 and cfg.grid == (4, 4)


def test_query_dispatch_unknown_provider():
    try:
        kairo.llm.query("hi", provider="nope")
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "nope" in str(exc)
