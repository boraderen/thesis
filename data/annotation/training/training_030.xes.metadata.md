# Rheon log: `training_030`

- **Events:** 5,208
- **Cases:** 868
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-02-26T16:51:34.643330+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden burstiness (inter_case)

- **Drift id:** `d01`
- **Change point:** 2020-02-02T00:19:26.247078+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **mean_preserved:** True
- **new_shape:** bimodal_pareto
- **old_shape:** exponential

### #2 · sudden reassignment (resource)

- **Drift id:** `d02`
- **Change point:** 2020-01-16T11:49:21.036462+00:00
- **Affected columns:** org:resource, org:role
- **activities:** all
- **reassignments:** [{'activity': 'a', 'old_dominant_resource': 'res_004', 'new_dominant_resource': 'res_006', 'post_drift_probability': 0.8}, {'activity': 'b', 'old_dominant_resource': 'res_007', 'new_dominant_resource': 'res_001', 'post_drift_probability': 0.8}, {'activity': 'c', 'old_dominant_resource': 'res_001', 'new_dominant_resource': 'res_002', 'post_drift_probability': 0.8}, {'activity': 'd', 'old_dominant_resource': 'res_002', 'new_dominant_resource': 'res_007', 'post_drift_probability': 0.8}, {'activity': 'e', 'old_dominant_resource': 'res_003', 'new_dominant_resource': 'res_002', 'post_drift_probability': 0.8}, {'activity': 'f', 'old_dominant_resource': 'res_008', 'new_dominant_resource': 'res_005', 'post_drift_probability': 0.8}]

### #3 · sudden reassignment (resource)

- **Drift id:** `d03`
- **Change point:** 2020-02-05T09:54:18.287367+00:00
- **Affected columns:** org:resource, org:role
- **activities:** all
- **reassignments:** [{'activity': 'a', 'old_dominant_resource': 'res_006', 'new_dominant_resource': 'res_001', 'post_drift_probability': 0.8}, {'activity': 'b', 'old_dominant_resource': 'res_001', 'new_dominant_resource': 'res_004', 'post_drift_probability': 0.8}, {'activity': 'c', 'old_dominant_resource': 'res_002', 'new_dominant_resource': 'res_003', 'post_drift_probability': 0.8}, {'activity': 'd', 'old_dominant_resource': 'res_007', 'new_dominant_resource': 'res_006', 'post_drift_probability': 0.8}, {'activity': 'e', 'old_dominant_resource': 'res_002', 'new_dominant_resource': 'res_005', 'post_drift_probability': 0.8}, {'activity': 'f', 'old_dominant_resource': 'res_005', 'new_dominant_resource': 'res_002', 'post_drift_probability': 0.8}]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 5208 |
| actual_num_traces | 868 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1029 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-02-26T16:51:34.643330+00:00 |
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
| num_resources | 8 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 800 |
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
