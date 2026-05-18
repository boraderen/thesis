# Driftify Usage Guide

Driftify generates synthetic XES event logs with injected concept drift. It is designed for experiments where you need a known ground truth: every generated log contains the full drift metadata embedded in the XES file (log-level attributes) and a human-readable Markdown summary written alongside the log.

The main entry points are the standalone scripts in this folder. You usually edit the constants at the top of one script, then run it.

## Quick Start

From the repository root:

```bash
uv run python scripts/generate_control_flow_log.py
```

Validate the generated XES file:

```bash
uv run python scripts/validate_log.py data/control-flow/control_flow_001.xes
```

The generator writes:

```text
<output_folder>/<log_name>.xes
<output_folder>/<log_name>.xes.metadata.md
```

The `.xes.metadata.md` file is a Markdown summary intended for humans. The XES file stores the machine-readable copy as log-level attributes (`drift_info`, `config_info`, `noise_info`) — use those when reading metadata programmatically.

## Available Generator Scripts

`generate_control_flow_log.py`

Generates logs where the process model changes. This is the CDLG-style drift perspective.

Supported subtypes:

```python
{"subtype": "tree_mutation", "drift_type": "sudden"}
```

The tree mutation can add activities, delete activities, move activities, and swap operators such as sequence/choice/parallel. The metadata records these changes as `activities_added`, `activities_deleted`, `activities_moved`, and `operator_swaps`. After trace generation, Driftify also records the real trace variant counts observed in the generated log.

`generate_resource_log.py`

Generates logs where resource behavior changes.

Supported subtypes:

```python
{"subtype": "reassignment", "drift_type": "sudden"}
{"subtype": "pool_size", "drift_type": "gradual_linear"}
{"subtype": "handover", "drift_type": "incremental"}
{"subtype": "workload_distribution", "drift_type": "recurring"}
```

What these mean:

- `reassignment`: an activity is mostly done by a different resource after drift.
- `pool_size`: the number of active resources changes.
- `handover`: dominant resource-to-resource handover changes between consecutive events.
- `workload_distribution`: work becomes more uniform or more concentrated on a few resources.

`generate_inter_case_log.py`

Generates logs where case-level timing or case mix changes.

Supported subtypes:

```python
{"subtype": "arrival_rate", "drift_type": "sudden"}
{"subtype": "burstiness", "drift_type": "gradual_exponential"}
{"subtype": "case_mix", "drift_type": "incremental"}
{"subtype": "concurrency", "drift_type": "sudden"}
```

What these mean:

- `arrival_rate`: cases arrive faster or slower.
- `burstiness`: arrivals become bursty, with quiet periods and sudden bursts.
- `case_mix`: the distribution of `case:case_type` changes.
- `concurrency`: this is not directly injected. It is an emergent effect from arrivals and event durations.

`generate_data_log.py`

Generates logs where data attributes change.

Supported subtypes:

```python
{"subtype": "numeric", "drift_type": "sudden"}
{"subtype": "categorical", "drift_type": "gradual_linear"}
```

What these mean:

- `numeric`: the `case:amount` mean and/or variance changes.
- `categorical`: the `case:region` distribution changes.

`generate_multi_perspective_log.py`

Combines multiple perspectives in one log. Each drift entry includes a `perspective` field:

```python
DRIFTS = [
    {"perspective": "control_flow", "subtype": "tree_mutation", "drift_type": "sudden"},
    {"perspective": "resource", "subtype": "reassignment", "drift_type": "gradual_linear"},
    {"perspective": "inter_case", "subtype": "arrival_rate", "drift_type": "sudden"},
    {"perspective": "data", "subtype": "numeric", "drift_type": "incremental"},
]
```

Drifts from different perspectives get independent change points. Drifts within the same perspective happen sequentially.

## Drift Types

All perspectives support these drift types:

`sudden`

A hard switch at the change point.

```python
{"subtype": "tree_mutation", "drift_type": "sudden"}
```

`gradual_linear`

Old and new behavior overlap during a transition window. The probability of using the new behavior increases linearly.

```python
{"subtype": "reassignment", "drift_type": "gradual_linear"}
```

`gradual_exponential`

Like gradual linear, but the new behavior starts slowly and rises faster near the end of the overlap window.

```python
{"subtype": "burstiness", "drift_type": "gradual_exponential"}
```

`incremental`

Creates several intermediate versions. Useful when the process should change step by step.

```python
{"subtype": "numeric", "drift_type": "incremental", "num_versions": 4}
```

`recurring`

Alternates between old and new behavior for a period.

```python
{"subtype": "workload_distribution", "drift_type": "recurring"}
```

## Important Configuration Knobs

Edit these at the top of a generator script.

### Output

```python
OUTPUT_PATH = "data/generated/"
NUM_LOGS = 10
GLOBAL_SEED = 42
```

- `OUTPUT_PATH`: folder where XES and metadata files are written.
- `NUM_LOGS`: how many logs to generate.
- `GLOBAL_SEED`: base seed. Log `001` uses this seed, log `002` uses `GLOBAL_SEED + 1`, etc.

### Log Size

```python
NUM_TRACES = 2000
MIN_TRACE_LENGTH = 5
MAX_TRACE_LENGTH = 50
AVG_TRACE_LENGTH = 15
TRACE_LENGTH_VARIANCE = 25
```

- `NUM_TRACES`: approximate number of cases. Arrival-rate drift can change the exact count.
- `MIN_TRACE_LENGTH` / `MAX_TRACE_LENGTH`: accepted trace length range.
- `AVG_TRACE_LENGTH`: target average trace length.
- `TRACE_LENGTH_VARIANCE`: target trace length spread.

Driftify now tries to choose process trees and playout traces whose lengths match these settings. This avoids the old behavior where many short traces were padded to exactly `MIN_TRACE_LENGTH`.

### Time Horizon

```python
HORIZON_MIN_DAYS = 30
HORIZON_MAX_DAYS = 365
```

The actual horizon is sampled between these values and scaled by the expected event count. Larger logs are spread over longer time ranges.

### Process Complexity

```python
MIN_ACTIVITIES = 6
MAX_ACTIVITIES = 20
SEQUENCE_WEIGHT = 0.70
CHOICE_WEIGHT = 0.20
PARALLEL_WEIGHT = 0.07
LOOP_WEIGHT = 0.02
OR_WEIGHT = 0.01
SILENT_TRANSITION_PROB = 0.05
DUPLICATE_ACTIVITY_PROB = 0.0
```

- `MIN_ACTIVITIES` / `MAX_ACTIVITIES`: number of visible activity labels in the process tree.
- `SEQUENCE_WEIGHT`: higher values usually make traces longer and more regular.
- `CHOICE_WEIGHT`: higher values usually create more alternative variants, often shorter traces.
- `PARALLEL_WEIGHT`: adds concurrent branches, which can increase variants.
- `LOOP_WEIGHT`: adds repetitions.
- `OR_WEIGHT`: adds inclusive-choice behavior.
- `SILENT_TRANSITION_PROB`: adds silent/skipped paths.
- `DUPLICATE_ACTIVITY_PROB`: allows duplicate activity labels. Usually keep this `0.0` for clean synthetic logs.

`TREE_DEPTH_MIN` and `TREE_DEPTH_MAX` exist in the config for readability, but the current PM4PY tree generator path does not use them as direct hard limits.

### Resources And Case Attributes

```python
NUM_RESOURCES = 20
NUM_ROLES = 5
NUM_CASE_TYPES = 3
REGIONS = ["DE-NRW", "DE-BY", "DE-HE", "DE-BW", "DE-BE"]
```

- Resources are named `res_001`, `res_002`, ...
- Roles are named `role_01`, `role_02`, ...
- Case types are named `type_01`, `type_02`, ...
- Regions are sampled from `REGIONS`.

There are no activity-role constraints. Any role can do any activity.

### Timing

```python
INTER_ARRIVAL_MEAN_MIN = 5
SERVICE_TIME_MEAN_MIN = 10
SERVICE_TIME_STD_MIN = 5
```

- `INTER_ARRIVAL_MEAN_MIN`: average time between case starts before inter-case drift.
- `SERVICE_TIME_MEAN_MIN`: average event duration.
- `SERVICE_TIME_STD_MIN`: event duration spread.

Each event has both `start_timestamp` and `time:timestamp`. The completion timestamp is `time:timestamp`.

### Noise

```python
NOISE_PROBABILITY = 0.05
NOISE_SIMILAR_VS_RANDOM = 0.5
```

- `NOISE_PROBABILITY`: fraction of traces that become noisy.
- `NOISE_SIMILAR_VS_RANDOM`: share of noisy traces that use similar-trace noise.

Noise types:

- Similar-trace noise: delete, swap, or duplicate 1-2 events in an existing valid trace.
- Fully random noise: generate a random activity trace with random length in the configured trace range.

## Control-Flow Similarity Knob

For control-flow drift, use `change_proportion` to control how different the pre-drift and post-drift process trees are.

```python
DRIFTS = [
    {"subtype": "tree_mutation", "drift_type": "sudden", "change_proportion": 0.10},
]
```

Suggested values:

- `0.05`: subtle drift
- `0.20`: medium drift
- `0.50`: strong drift

Higher values cause more tree mutations, so traces before and after drift become less similar.

## Output Schema

Generated logs use PM4PY/XES-friendly names:

```text
event:id
case:concept:name
concept:name
start_timestamp
time:timestamp
event:duration_min
org:resource
org:role
case:case_type
case:amount
case:region
```

Notes:

- `concept:name` is the activity.
- `case:concept:name` is the case id.
- `time:timestamp` is the event completion time.
- `start_timestamp` is the event start time.
- `event:duration_min` is `(time:timestamp - start_timestamp)` in minutes.
- `case:*` attributes are case-level attributes.

## Metadata And Gold Standard

Each drift is stored with:

```text
drift_id
perspective
subtype
drift_type
change_point_timestamp
overlap_window_start
overlap_window_end
affected_columns
change_details
```

The top-level `config` section also contains generated-log measurements:

```text
actual_num_traces
actual_num_events
num_trace_variants
num_trace_variants_before_noise
```

`num_trace_variants` is the number of unique activity sequences actually present in the final log. This is the value to compare with PM4PY's variant count. `num_trace_variants_before_noise` is counted before noise injection, so it can differ when noisy traces add or remove variants.

For control-flow drifts, `change_details` contains the exact activities added/deleted/moved and operator swaps. It also contains observed variant counts:

```text
variant_count_before
variant_count_after
variant_counts_by_version
```

These are observed trace variants from the generated cases, not process-tree structural estimates.

For a quick human-readable view, open the `.xes.metadata.md` sidecar in any Markdown viewer.

To read the metadata programmatically, pull the JSON-encoded log-level attributes:

```python
import json
from pm4py.objects.log.importer.xes import importer as xes_importer

log = xes_importer.apply("data/control-flow/control_flow_001.xes")
drifts = json.loads(log.attributes["drift_info"])
config = json.loads(log.attributes["config_info"])
noise = json.loads(log.attributes["noise_info"])
print(drifts)
```

## Validation

Run:

```bash
uv run python scripts/validate_log.py path/to/log.xes
```

The validator checks:

- required schema columns
- timestamp order inside each case
- `start_timestamp <= time:timestamp`
- duration consistency
- drift change points inside the log time range
- resource, activity, role, case type, and region values
- XES well-formedness and PM4PY round-trip import
- basic distribution stability warnings for periods without drift

For machine-readable output:

```bash
uv run python scripts/validate_log.py path/to/log.xes --json
```

## Common Recipes

Generate one clear sudden control-flow drift:

```python
NUM_LOGS = 1
NUM_TRACES = 10000
NOISE_PROBABILITY = 0.005

DRIFTS = [
    {"subtype": "tree_mutation", "drift_type": "sudden", "change_proportion": 0.20},
]
```

Generate a smoother control-flow drift:

```python
DRIFTS = [
    {"subtype": "tree_mutation", "drift_type": "gradual_linear", "change_proportion": 0.20},
]
```

Generate multiple resource drifts:

```python
DRIFTS = [
    {"subtype": "reassignment", "drift_type": "sudden"},
    {"subtype": "pool_size", "drift_type": "gradual_linear"},
    {"subtype": "handover", "drift_type": "incremental", "num_versions": 4},
]
```

Generate a multi-perspective log:

```python
DRIFTS = [
    {"perspective": "control_flow", "subtype": "tree_mutation", "drift_type": "sudden", "change_proportion": 0.15},
    {"perspective": "resource", "subtype": "workload_distribution", "drift_type": "gradual_linear"},
    {"perspective": "data", "subtype": "numeric", "drift_type": "incremental"},
]
```

## Troubleshooting

Many traces have the same length.

Increase tree and trace variation:

```python
TRACE_LENGTH_VARIANCE = 35
MIN_ACTIVITIES = 14
MAX_ACTIVITIES = 26
CHOICE_WEIGHT = 0.25
LOOP_WEIGHT = 0.05
SILENT_TRANSITION_PROB = 0.0
```

Too many short traces.

Raise `MIN_TRACE_LENGTH`, raise `AVG_TRACE_LENGTH`, increase `SEQUENCE_WEIGHT`, and reduce `CHOICE_WEIGHT` or `SILENT_TRANSITION_PROB`.

Generated logs are too slow.

Reduce `NUM_TRACES`, `MAX_ACTIVITIES`, `MAX_TRACE_LENGTH`, or `NUM_LOGS`. Complex trees with many choices/parallel branches can be slower because there are many possible variants.

I see `case:case:*` columns after importing with PM4PY.

Regenerate the logs with the current writer. The writer avoids double prefixing by storing trace attributes as raw XES trace attributes and letting PM4PY add the `case:` prefix during import.