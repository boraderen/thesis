# Rheon log: `testing_002`

- **Events:** 8,944
- **Cases:** 1,500
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-04-19T12:00:00+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden tree_mutation (control_flow)

- **Drift id:** `d01`
- **Change point:** 2020-02-13T01:51:06.392164+00:00
- **Affected columns:** concept:name
- **activities_added:** []
- **activities_deleted:** []
- **activities_moved:** ['b', 'a', 'c', 'e', 'd', 'f', 'd', 'b', 'b']
- **operator_swaps:** [{'from': 'sequence', 'to': 'loop', 'affected_activities': 'b,a,c,e,d,f'}]
- **variant_count_after:** 1
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 1]

### #2 · sudden tree_mutation (control_flow)

- **Drift id:** `d02`
- **Change point:** 2020-03-16T08:49:08.388614+00:00
- **Affected columns:** concept:name
- **activities_added:** []
- **activities_deleted:** []
- **activities_moved:** ['d', 'b', 'a', 'c', 'e', 'f', 'c', 'd', 'b', 'a', 'c', 'e', 'f']
- **operator_swaps:** [{'from': 'sequence', 'to': 'choice', 'affected_activities': 'd,b,a,c,e,f'}, {'from': 'loop', 'to': 'or', 'affected_activities': 'd,b,a,c,e,f'}]
- **variant_count_after:** 12
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 12]

### #3 · sudden case_mix (inter_case)

- **Drift id:** `d03`
- **Change point:** 2020-02-28T15:39:07.245572+00:00
- **Affected columns:** case:case_type
- **dominant_case_type:** type_02
- **new_case_type_distribution:** [0.15, 0.7, 0.15]
- **old_case_type_distribution:** [0.3333, 0.3333, 0.3333]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 8944 |
| actual_num_traces | 1500 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1031 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-04-19T12:00:00+00:00 |
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
| num_resources | 8 |
| num_roles | 3 |
| num_trace_variants | 14 |
| num_trace_variants_before_noise | 14 |
| num_traces | 1500 |
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
