# Rheon log: `training_018`

- **Events:** 9,596
- **Cases:** 1,500
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-04-17T21:47:19.035662+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden workload_distribution (resource)

- **Drift id:** `d01`
- **Change point:** 2020-03-01T21:19:50.291364+00:00
- **Affected columns:** org:resource, org:role
- **affected_resources:** ['res_001', 'res_002', 'res_003', 'res_004', 'res_005', 'res_006', 'res_007', 'res_008', 'res_009', 'res_010']
- **heavy_resource_share:** 0.7
- **heavy_resources:** ['res_002', 'res_005', 'res_008', 'res_009']
- **probability_changes:** {'res_001': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_002': {'old_probability': 0.0222, 'new_probability': 0.175}, 'res_003': {'old_probability': 0.1519, 'new_probability': 0.05}, 'res_004': {'old_probability': 0.2815, 'new_probability': 0.05}, 'res_005': {'old_probability': 0.1519, 'new_probability': 0.175}, 'res_006': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_007': {'old_probability': 0.0222, 'new_probability': 0.05}, 'res_008': {'old_probability': 0.1519, 'new_probability': 0.175}, 'res_009': {'old_probability': 0.1519, 'new_probability': 0.175}, 'res_010': {'old_probability': 0.0222, 'new_probability': 0.05}}
- **resources:** all

### #2 · sudden tree_mutation (control_flow)

- **Drift id:** `d02`
- **Change point:** 2020-03-05T03:40:13.644964+00:00
- **Affected columns:** concept:name
- **activities_added:** []
- **activities_deleted:** []
- **activities_moved:** ['c', 'a', 'b', 'd', 'f', 'e', 'c', 'a', 'b', 'd', 'f', 'e']
- **operator_swaps:** [{'from': 'sequence', 'to': 'loop', 'affected_activities': 'c,a,b,d,f,e'}, {'from': 'sequence', 'to': 'loop', 'affected_activities': 'c,a,b,d,f,e'}]
- **variant_count_after:** 1
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 1]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 9596 |
| actual_num_traces | 1500 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1017 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-04-17T21:47:19.035662+00:00 |
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
| num_resources | 10 |
| num_roles | 3 |
| num_trace_variants | 2 |
| num_trace_variants_before_noise | 2 |
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
