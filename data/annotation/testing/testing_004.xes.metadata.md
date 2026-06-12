# Rheon log: `testing_004`

- **Events:** 10,896
- **Cases:** 2,000
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-05-17T09:45:56.709206+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden categorical (data)

- **Drift id:** `d01`
- **Change point:** 2020-03-19T22:53:40.401525+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-NRW
- **new_region_distribution:** [0.7, 0.075, 0.075, 0.075, 0.075]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

### #2 · sudden pool_size (resource)

- **Drift id:** `d02`
- **Change point:** 2020-03-21T06:19:53.969212+00:00
- **Affected columns:** org:resource, org:role
- **active_resources:** ['res_001', 'res_003', 'res_005', 'res_006']
- **new_pool_size:** 4
- **old_pool_size:** 6

### #3 · sudden tree_mutation (control_flow)

- **Drift id:** `d03`
- **Change point:** 2020-03-13T03:46:26.503349+00:00
- **Affected columns:** concept:name
- **activities_added:** []
- **activities_deleted:** []
- **activities_moved:** ['a', 'd', 'f', 'c', 'b', 'e', 'a', 'd', 'f', 'c', 'b', 'e']
- **operator_swaps:** [{'from': 'sequence', 'to': 'choice', 'affected_activities': 'a,d,f,c,b,e'}, {'from': 'sequence', 'to': 'or', 'affected_activities': 'a,d,f,c,b,e'}]
- **variant_count_after:** 3
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 3]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 10896 |
| actual_num_traces | 2000 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1033 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-05-17T09:45:56.709206+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 240 |
| horizon_start | 2020-01-01T00:00:00+00:00 |
| inter_arrival_mean_min | 45 |
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
| num_trace_variants | 4 |
| num_trace_variants_before_noise | 4 |
| num_traces | 2000 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 15 |
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
