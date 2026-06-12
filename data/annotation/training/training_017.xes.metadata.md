# Rheon log: `training_017`

- **Events:** 4,788
- **Cases:** 798
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-02-20T08:54:24.129142+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden handover (resource)

- **Drift id:** `d01`
- **Change point:** 2020-01-25T14:35:52.179422+00:00
- **Affected columns:** org:resource, org:role
- **new_dominant_target:** res_002
- **old_dominant_target:** res_001
- **post_drift_probability:** 0.8
- **source_resource:** res_002

### #2 · sudden burstiness (inter_case)

- **Drift id:** `d02`
- **Change point:** 2020-01-24T17:03:44.476638+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **mean_preserved:** True
- **new_shape:** bimodal_pareto
- **old_shape:** exponential

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 4788 |
| actual_num_traces | 798 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1016 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-02-20T08:54:24.129142+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 300 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 45 |
| loop_weight | 0.0 |
| max_activities | 6 |
| max_trace_length | 8 |
| min_activities | 6 |
| min_trace_length | 4 |
| noise_probability | 0 |
| noise_similar_vs_random | 0 |
| num_case_types | 3 |
| num_resources | 6 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 800 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 15 |
| service_time_std_min | 9 |
| silent_transition_prob | 0.0 |
| start_timestamp | 2020-01-01T00:00:00+00:00 |
| trace_length_variance | 25 |
| tree_depth_max | 6 |
| tree_depth_min | 3 |
| tree_generation_attempts | 32 |

## Noise

| Setting | Value |
| --- | --- |
| noise_probability | 0.0 |
| noisy_traces | 0 |
