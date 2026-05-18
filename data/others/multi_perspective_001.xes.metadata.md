# Driftify log: `multi_perspective_001`

- **Events:** 55,860
- **Cases:** 9,310
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2021-02-12T18:10:43.368542+00:00
- **Injected drifts:** 4

## Injected drifts

### #1 · sudden tree_mutation (control_flow)

- **Drift id:** `d01`
- **Change point:** 2020-08-24T20:26:13.314252+00:00
- **Affected columns:** concept:name
- **activities_added:** ['a']
- **activities_deleted:** ['a']
- **activities_moved:** ['f']
- **operator_swaps:** []
- **variant_count_after:** 1
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 1]

### #2 · sudden reassignment (resource)

- **Drift id:** `d02`
- **Change point:** 2020-08-14T21:59:40.325111+00:00
- **Affected columns:** org:resource, org:role
- **activity:** f
- **new_dominant_resource:** res_007
- **old_dominant_resource:** res_010
- **post_drift_probability:** 0.8

### #3 · sudden arrival_rate (inter_case)

- **Drift id:** `d03`
- **Change point:** 2020-06-30T21:56:09.750003+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **new_interarrival_mean_multiplier:** 0.5
- **old_interarrival_mean_multiplier:** 1.0

### #4 · sudden numeric (data)

- **Drift id:** `d04`
- **Change point:** 2020-07-07T01:00:30.474511+00:00
- **Affected columns:** case:amount
- **new_amount_mean:** 1450.0
- **new_amount_std:** 175.0
- **old_amount_mean:** 1000.0
- **old_amount_std:** 250.0

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 55860 |
| actual_num_traces | 9310 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 7 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2021-02-12T18:10:43.368542+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 300 |
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
| num_logs | 1 |
| num_resources | 10 |
| num_roles | 3 |
| num_trace_variants | 2 |
| num_trace_variants_before_noise | 2 |
| num_traces | 6000 |
| or_weight | 0.01 |
| output_path | data/others/ |
| parallel_weight | 0.01 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 15 |
| service_time_std_min | 7 |
| silent_transition_prob | 0.0 |
| start_timestamp | 2020-01-01T00:00:00+00:00 |
| trace_length_variance | 25 |
| tree_depth_max | 2 |
| tree_depth_min | 2 |
| tree_generation_attempts | 32 |

## Noise

| Setting | Value |
| --- | --- |
| noise_probability | 0.0 |
| noisy_traces | 0 |
