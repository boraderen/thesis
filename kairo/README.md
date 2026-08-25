# kairo

State-based process monitoring and concept-drift detection for traditional
event logs — with textual abstractions and LLM access built in.

kairo computes three complementary notions of state from an event log
(intra-case, resource, inter-case), follows how state occupancy evolves over
calendar windows, and turns changes in those signals into drift indicators.
Every computed object has an `abstract_*` function that renders it as text, so
a language model — hosted or local — can read and reason about a full analysis.

```python
import kairo

log = kairo.read_log("log.xes")
result = kairo.run_intra_case(log, kairo.IntraConfig(clustering="som", grid=(3, 3)))

kairo.plot_state_grid(result.states).show()
print(kairo.abstract_result(result))

answer = kairo.ask(
    "Where does the process drift, and what characterises the change?",
    result=result, log=log,
    executor=kairo.anthropic_query,   # or openai_query / google_query / local_query
)
```

Open `index.html` for the full documentation. It ships inside the thesis repository — `uv sync` from the repo root installs it.
