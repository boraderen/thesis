# Rheon log: `training_022`

- **Events:** 9,000
- **Cases:** 1,500
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-03-15T18:41:06.021056+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden numeric (data)

- **Drift id:** `d01`
- **Change point:** 2020-02-01T02:53:40.437465+00:00
- **Affected columns:** case:amount
- **new_amount_mean:** 1450.0
- **new_amount_std:** 400.0
- **old_amount_mean:** 1000.0
- **old_amount_std:** 250.0

### #2 · sudden concurrency (inter_case)

- **Drift id:** `d02`
- **Change point:** 2020-02-14T17:44:48.815383+00:00
- **direct_injection:** False
- **note:** Concurrency is emergent from arrivals and service times.

### #3 · sudden service_time (resource)

- **Drift id:** `d03`
- **Change point:** 2020-02-01T14:23:48.972810+00:00
- **Affected columns:** org:resource, org:role
- **multiplier_changes:** {'res_001': {'old_multiplier': 1.0, 'new_multiplier': 1.1666}, 'res_002': {'old_multiplier': 1.0, 'new_multiplier': 2.6574}, 'res_003': {'old_multiplier': 1.0, 'new_multiplier': 0.5261}, 'res_004': {'old_multiplier': 1.0, 'new_multiplier': 1.9456}, 'res_005': {'old_multiplier': 1.0, 'new_multiplier': 1.9012}, 'res_006': {'old_multiplier': 1.0, 'new_multiplier': 2.346}, 'res_007': {'old_multiplier': 1.0, 'new_multiplier': 2.4552}, 'res_008': {'old_multiplier': 1.0, 'new_multiplier': 2.7865}}
- **note:** multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster
- **resources:** all

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 9000 |
| actual_num_traces | 1500 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1021 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-03-15T18:41:06.021056+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 180 |
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
