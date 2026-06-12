# Rheon log: `training_024`

- **Events:** 9,000
- **Cases:** 1,500
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-03-13T19:22:25.012457+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden service_time (resource)

- **Drift id:** `d01`
- **Change point:** 2020-02-12T07:32:17.358936+00:00
- **Affected columns:** org:resource, org:role
- **multiplier_changes:** {'res_001': {'old_multiplier': 1.0, 'new_multiplier': 2.8095}, 'res_002': {'old_multiplier': 1.0, 'new_multiplier': 1.4162}, 'res_003': {'old_multiplier': 1.0, 'new_multiplier': 1.1869}, 'res_004': {'old_multiplier': 1.0, 'new_multiplier': 0.9356}, 'res_005': {'old_multiplier': 1.0, 'new_multiplier': 1.1335}, 'res_006': {'old_multiplier': 1.0, 'new_multiplier': 0.8568}, 'res_007': {'old_multiplier': 1.0, 'new_multiplier': 2.7888}, 'res_008': {'old_multiplier': 1.0, 'new_multiplier': 2.5149}}
- **note:** multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster
- **resources:** all

### #2 · sudden categorical (data)

- **Drift id:** `d02`
- **Change point:** 2020-02-12T08:49:56.408434+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-BY
- **new_region_distribution:** [0.075, 0.7, 0.075, 0.075, 0.075]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 9000 |
| actual_num_traces | 1500 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1023 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-03-13T19:22:25.012457+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 180 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 30 |
| loop_weight | 0.0 |
| max_activities | 6 |
| max_trace_length | 8 |
| min_activities | 6 |
| min_trace_length | 4 |
| noise_probability | 0 |
| noise_similar_vs_random | 0 |
| num_case_types | 3 |
| num_resources | 8 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 1500 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 15 |
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
