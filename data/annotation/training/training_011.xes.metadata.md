# Rheon log: `training_011`

- **Events:** 7,200
- **Cases:** 1,200
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-03-28T14:24:00+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden numeric (data)

- **Drift id:** `d01`
- **Change point:** 2020-02-14T10:15:20.137934+00:00
- **Affected columns:** case:amount
- **new_amount_mean:** 1450.0
- **new_amount_std:** 400.0
- **old_amount_mean:** 1000.0
- **old_amount_std:** 250.0

### #2 · sudden service_time (resource)

- **Drift id:** `d02`
- **Change point:** 2020-02-04T01:52:59.310575+00:00
- **Affected columns:** org:resource, org:role
- **multiplier_changes:** {'res_001': {'old_multiplier': 1.0, 'new_multiplier': 1.8572}, 'res_002': {'old_multiplier': 1.0, 'new_multiplier': 1.0383}, 'res_003': {'old_multiplier': 1.0, 'new_multiplier': 2.9292}, 'res_004': {'old_multiplier': 1.0, 'new_multiplier': 1.8795}, 'res_005': {'old_multiplier': 1.0, 'new_multiplier': 2.7779}, 'res_006': {'old_multiplier': 1.0, 'new_multiplier': 2.4584}}
- **note:** multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster
- **resources:** all

### #3 · sudden workload_distribution (resource)

- **Drift id:** `d03`
- **Change point:** 2020-03-05T00:33:52.224338+00:00
- **Affected columns:** org:resource, org:role
- **affected_resources:** ['res_001', 'res_002', 'res_003', 'res_004', 'res_005', 'res_006']
- **heavy_resource_share:** 0.7
- **heavy_resources:** ['res_005', 'res_006']
- **probability_changes:** {'res_001': {'old_probability': 0.1667, 'new_probability': 0.075}, 'res_002': {'old_probability': 0.5467, 'new_probability': 0.075}, 'res_003': {'old_probability': 0.04, 'new_probability': 0.075}, 'res_004': {'old_probability': 0.04, 'new_probability': 0.075}, 'res_005': {'old_probability': 0.04, 'new_probability': 0.35}, 'res_006': {'old_probability': 0.1667, 'new_probability': 0.35}}
- **resources:** all

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 7200 |
| actual_num_traces | 1200 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1010 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-03-28T14:24:00+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 365 |
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
| num_resources | 6 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 1200 |
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
