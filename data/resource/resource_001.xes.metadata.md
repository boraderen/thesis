# Rheon log: `resource_001`

- **Events:** 6,000
- **Cases:** 1,000
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-03-14T00:00:00+00:00
- **Injected drifts:** 5

## Injected drifts

### #1 · sudden pool_size (resource)

- **Drift id:** `d01`
- **Change point:** 2020-02-06T12:00:00+00:00
- **Affected columns:** org:resource, org:role
- **active_resources:** ['res_001', 'res_002', 'res_003', 'res_005', 'res_006']
- **new_pool_size:** 5
- **old_pool_size:** 8

### #2 · sudden workload_distribution (resource)

- **Drift id:** `d02`
- **Change point:** 2020-02-06T12:00:00+00:00
- **Affected columns:** org:resource, org:role
- **affected_resources:** ['res_001', 'res_002', 'res_003', 'res_005', 'res_006']
- **heavy_resource_share:** 0.7
- **heavy_resources:** ['res_002', 'res_006']
- **probability_changes:** {'res_001': {'old_probability': 0.2281, 'new_probability': 0.1}, 'res_002': {'old_probability': 0.3687, 'new_probability': 0.35}, 'res_003': {'old_probability': 0.0875, 'new_probability': 0.1}, 'res_005': {'old_probability': 0.0875, 'new_probability': 0.1}, 'res_006': {'old_probability': 0.2281, 'new_probability': 0.35}}
- **resources:** all

### #3 · sudden reassignment (resource)

- **Drift id:** `d03`
- **Change point:** 2020-02-06T12:00:00+00:00
- **Affected columns:** org:resource, org:role
- **activities:** ['a', 'c']
- **reassignments:** [{'activity': 'a', 'old_dominant_resource': 'res_002', 'new_dominant_resource': 'res_001', 'post_drift_probability': 0.8}, {'activity': 'c', 'old_dominant_resource': 'res_002', 'new_dominant_resource': 'res_005', 'post_drift_probability': 0.8}]

### #4 · sudden service_time (resource)

- **Drift id:** `d04`
- **Change point:** 2020-02-06T12:00:00+00:00
- **Affected columns:** org:resource, org:role
- **multiplier_changes:** {'res_001': {'old_multiplier': 1.0, 'new_multiplier': 2.5873}, 'res_002': {'old_multiplier': 1.0, 'new_multiplier': 2.0272}, 'res_003': {'old_multiplier': 1.0, 'new_multiplier': 2.3028}, 'res_005': {'old_multiplier': 1.0, 'new_multiplier': 0.547}, 'res_006': {'old_multiplier': 1.0, 'new_multiplier': 1.7611}}
- **note:** multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster
- **resources:** all

### #5 · sudden handover (resource)

- **Drift id:** `d05`
- **Change point:** 2020-02-06T12:00:00+00:00
- **Affected columns:** org:resource, org:role
- **new_dominant_target:** res_005
- **old_dominant_target:** res_001
- **post_drift_probability:** 0.8
- **source_resource:** res_005

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 6000 |
| actual_num_traces | 1000 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 7 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-03-14T00:00:00+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 365 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 30 |
| loop_weight | 0.0 |
| max_activities | 6 |
| max_trace_length | 7 |
| min_activities | 6 |
| min_trace_length | 5 |
| noise_probability | 0 |
| noise_similar_vs_random | 0 |
| num_case_types | 3 |
| num_resources | 8 |
| num_roles | 2 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 1000 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 15 |
| service_time_std_min | 5 |
| silent_transition_prob | 0.0 |
| start_timestamp | 2020-01-01T00:00:00+00:00 |
| trace_length_variance | 9 |
| tree_depth_max | 6 |
| tree_depth_min | 3 |
| tree_generation_attempts | 32 |

## Noise

| Setting | Value |
| --- | --- |
| noise_probability | 0.0 |
| noisy_traces | 0 |
