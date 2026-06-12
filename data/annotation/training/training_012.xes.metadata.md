# Rheon log: `training_012`

- **Events:** 17,682
- **Cases:** 2,947
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-05-26T00:00:00+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden arrival_rate (inter_case)

- **Drift id:** `d01`
- **Change point:** 2020-03-15T20:22:37.681266+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **new_interarrival_mean_multiplier:** 0.5
- **old_interarrival_mean_multiplier:** 1.0

### #2 · sudden workload_distribution (resource)

- **Drift id:** `d02`
- **Change point:** 2020-03-05T14:35:24.627716+00:00
- **Affected columns:** org:resource, org:role
- **affected_resources:** ['res_001', 'res_002', 'res_003', 'res_004', 'res_005', 'res_006', 'res_007', 'res_008', 'res_009', 'res_010']
- **heavy_resource_share:** 0.7
- **heavy_resources:** ['res_004', 'res_005', 'res_006', 'res_009']
- **probability_changes:** {'res_001': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_002': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_003': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_004': {'old_probability': 0.0222, 'new_probability': 0.175}, 'res_005': {'old_probability': 0.0222, 'new_probability': 0.175}, 'res_006': {'old_probability': 0.0222, 'new_probability': 0.175}, 'res_007': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_008': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_009': {'old_probability': 0.5407, 'new_probability': 0.175}, 'res_010': {'old_probability': 0.2815, 'new_probability': 0.05}}
- **resources:** all

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 17682 |
| actual_num_traces | 2947 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1011 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-05-26T00:00:00+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 365 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 20 |
| loop_weight | 0.0 |
| max_activities | 6 |
| max_trace_length | 8 |
| min_activities | 6 |
| min_trace_length | 4 |
| noise_probability | 0 |
| noise_similar_vs_random | 0 |
| num_case_types | 3 |
| num_resources | 10 |
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
