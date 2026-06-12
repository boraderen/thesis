# Rheon log: `training_005`

- **Events:** 4,612
- **Cases:** 800
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-02-08T10:01:21.422336+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden tree_mutation (control_flow)

- **Drift id:** `d01`
- **Change point:** 2020-01-17T08:48:34.335422+00:00
- **Affected columns:** concept:name
- **activities_added:** []
- **activities_deleted:** []
- **activities_moved:** ['b', 'c', 'f', 'a', 'e', 'd', 'b', 'd', 'c', 'd', 'f', 'a', 'e', 'b']
- **operator_swaps:** [{'from': 'sequence', 'to': 'loop', 'affected_activities': 'c,f,a,e,d,b'}, {'from': 'sequence', 'to': 'or', 'affected_activities': 'c,d,f,a,e,b'}]
- **variant_count_after:** 10
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 10]

### #2 · sudden categorical (data)

- **Drift id:** `d02`
- **Change point:** 2020-01-17T10:20:41.638144+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-BY
- **new_region_distribution:** [0.075, 0.7, 0.075, 0.075, 0.075]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 4612 |
| actual_num_traces | 800 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1004 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-02-08T10:01:21.422336+00:00 |
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
| num_trace_variants | 11 |
| num_trace_variants_before_noise | 11 |
| num_traces | 800 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 10 |
| service_time_std_min | 5 |
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
