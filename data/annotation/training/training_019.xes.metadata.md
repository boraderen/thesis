# Rheon log: `training_019`

- **Events:** 7,200
- **Cases:** 1,200
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-03-16T10:03:44.364297+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden pool_size (resource)

- **Drift id:** `d01`
- **Change point:** 2020-02-07T00:43:32.828479+00:00
- **Affected columns:** org:resource, org:role
- **active_resources:** ['res_001', 'res_003', 'res_004', 'res_006']
- **new_pool_size:** 4
- **old_pool_size:** 6

### #2 · sudden categorical (data)

- **Drift id:** `d02`
- **Change point:** 2020-02-13T07:31:25.751559+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-BE
- **new_region_distribution:** [0.075, 0.075, 0.075, 0.075, 0.7]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 7200 |
| actual_num_traces | 1200 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1018 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-03-16T10:03:44.364297+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 240 |
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
| num_roles | 2 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 1200 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 20 |
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
