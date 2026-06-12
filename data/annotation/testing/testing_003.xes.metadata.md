# Rheon log: `testing_003`

- **Events:** 11,964
- **Cases:** 1,994
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-05-22T19:42:35.150141+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden workload_distribution (resource)

- **Drift id:** `d01`
- **Change point:** 2020-03-04T19:58:26.553733+00:00
- **Affected columns:** org:resource, org:role
- **affected_resources:** ['res_001', 'res_002', 'res_003', 'res_004', 'res_005', 'res_006']
- **heavy_resource_share:** 0.7
- **heavy_resources:** ['res_004', 'res_006']
- **probability_changes:** {'res_001': {'old_probability': 0.2933, 'new_probability': 0.075}, 'res_002': {'old_probability': 0.2933, 'new_probability': 0.075}, 'res_003': {'old_probability': 0.04, 'new_probability': 0.075}, 'res_004': {'old_probability': 0.1667, 'new_probability': 0.35}, 'res_005': {'old_probability': 0.1667, 'new_probability': 0.075}, 'res_006': {'old_probability': 0.04, 'new_probability': 0.35}}
- **resources:** all

### #2 · sudden burstiness (inter_case)

- **Drift id:** `d02`
- **Change point:** 2020-03-26T14:13:14.642953+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **mean_preserved:** True
- **new_shape:** bimodal_pareto
- **old_shape:** exponential

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 11964 |
| actual_num_traces | 1994 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1032 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-05-22T19:42:35.150141+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 300 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 20 |
| loop_weight | 0.0 |
| max_activities | 6 |
| max_trace_length | 8 |
| min_activities | 5 |
| min_trace_length | 4 |
| noise_probability | 0 |
| noise_similar_vs_random | 0 |
| num_case_types | 3 |
| num_resources | 6 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 2000 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 10 |
| service_time_std_min | 7 |
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
