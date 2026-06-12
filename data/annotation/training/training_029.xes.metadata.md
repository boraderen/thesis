# Rheon log: `training_029`

- **Events:** 9,000
- **Cases:** 1,500
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-04-19T12:00:00+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden reassignment (resource)

- **Drift id:** `d01`
- **Change point:** 2020-02-22T06:05:24.873890+00:00
- **Affected columns:** org:resource, org:role
- **activities:** all
- **reassignments:** [{'activity': 'a', 'old_dominant_resource': 'res_004', 'new_dominant_resource': 'res_006', 'post_drift_probability': 0.8}, {'activity': 'b', 'old_dominant_resource': 'res_006', 'new_dominant_resource': 'res_003', 'post_drift_probability': 0.8}, {'activity': 'c', 'old_dominant_resource': 'res_002', 'new_dominant_resource': 'res_005', 'post_drift_probability': 0.8}, {'activity': 'd', 'old_dominant_resource': 'res_005', 'new_dominant_resource': 'res_003', 'post_drift_probability': 0.8}, {'activity': 'e', 'old_dominant_resource': 'res_003', 'new_dominant_resource': 'res_002', 'post_drift_probability': 0.8}, {'activity': 'f', 'old_dominant_resource': 'res_005', 'new_dominant_resource': 'res_001', 'post_drift_probability': 0.8}]

### #2 · sudden categorical (data)

- **Drift id:** `d02`
- **Change point:** 2020-02-28T14:08:38.661137+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-HE
- **new_region_distribution:** [0.075, 0.075, 0.7, 0.075, 0.075]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 9000 |
| actual_num_traces | 1500 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1028 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-04-19T12:00:00+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 365 |
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
| num_roles | 2 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 1500 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 20 |
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
