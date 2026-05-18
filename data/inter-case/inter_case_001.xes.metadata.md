# Driftify log: `inter_case_001`

- **Events:** 63,972
- **Cases:** 10,662
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2021-02-12T18:10:43.368542+00:00
- **Injected drifts:** 4

## Injected drifts

### #1 · sudden arrival_rate (inter_case)

- **Drift id:** `d01`
- **Change point:** 2020-03-30T22:14:21.541079+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **new_interarrival_mean_multiplier:** 0.5
- **old_interarrival_mean_multiplier:** 1.0

### #2 · sudden burstiness (inter_case)

- **Drift id:** `d02`
- **Change point:** 2020-06-21T12:26:00.803753+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **mean_preserved:** True
- **new_shape:** bimodal_pareto
- **old_shape:** exponential

### #3 · sudden case_mix (inter_case)

- **Drift id:** `d03`
- **Change point:** 2020-08-24T06:26:45.247418+00:00
- **Affected columns:** case:case_type
- **dominant_case_type:** type_02
- **new_case_type_distribution:** [0.15, 0.7, 0.15]
- **old_case_type_distribution:** [0.3333, 0.3333, 0.3333]

### #4 · sudden concurrency (inter_case)

- **Drift id:** `d04`
- **Change point:** 2020-11-16T11:18:38.210930+00:00
- **direct_injection:** False
- **note:** Concurrency is emergent from arrivals and service times.

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 63972 |
| actual_num_traces | 10662 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 7 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2021-02-12T18:10:43.368542+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 300 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 30 |
| loop_weight | 0.0 |
| max_activities | 6 |
| max_trace_length | 8 |
| min_activities | 5 |
| min_trace_length | 4 |
| noise_probability | 0 |
| noise_similar_vs_random | 0 |
| num_case_types | 3 |
| num_logs | 1 |
| num_resources | 10 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 6000 |
| or_weight | 0.01 |
| output_path | data/timing-arrival/ |
| parallel_weight | 0.01 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 15 |
| service_time_std_min | 7 |
| silent_transition_prob | 0.0 |
| start_timestamp | 2020-01-01T00:00:00+00:00 |
| trace_length_variance | 25 |
| tree_depth_max | 2 |
| tree_depth_min | 2 |
| tree_generation_attempts | 32 |

## Noise

| Setting | Value |
| --- | --- |
| noise_probability | 0.0 |
| noisy_traces | 0 |
