# Rheon log: `training_015`

- **Events:** 11,437
- **Cases:** 2,000
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-05-11T07:46:01.117245+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden tree_mutation (control_flow)

- **Drift id:** `d01`
- **Change point:** 2020-03-08T21:54:22.419526+00:00
- **Affected columns:** concept:name
- **activities_added:** []
- **activities_deleted:** []
- **activities_moved:** ['a', 'd', 'c', 'b', 'e', 'f', 'a', 'd', 'c', 'b', 'e', 'f']
- **operator_swaps:** [{'from': 'sequence', 'to': 'or', 'affected_activities': 'a,d,c,b,e,f'}, {'from': 'sequence', 'to': 'or', 'affected_activities': 'a,d,c,b,e,f'}]
- **variant_count_after:** 22
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 22]

### #2 · sudden service_time (resource)

- **Drift id:** `d02`
- **Change point:** 2020-03-17T02:44:31.041024+00:00
- **Affected columns:** org:resource, org:role
- **multiplier_changes:** {'res_001': {'old_multiplier': 1.0, 'new_multiplier': 1.806}, 'res_002': {'old_multiplier': 1.0, 'new_multiplier': 0.8847}, 'res_003': {'old_multiplier': 1.0, 'new_multiplier': 2.1811}, 'res_004': {'old_multiplier': 1.0, 'new_multiplier': 2.6844}, 'res_005': {'old_multiplier': 1.0, 'new_multiplier': 0.6829}, 'res_006': {'old_multiplier': 1.0, 'new_multiplier': 1.6685}, 'res_007': {'old_multiplier': 1.0, 'new_multiplier': 2.8828}, 'res_008': {'old_multiplier': 1.0, 'new_multiplier': 2.3126}, 'res_009': {'old_multiplier': 1.0, 'new_multiplier': 1.1289}, 'res_010': {'old_multiplier': 1.0, 'new_multiplier': 0.6543}}
- **note:** multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster
- **resources:** all

### #3 · sudden categorical (data)

- **Drift id:** `d03`
- **Change point:** 2020-03-19T14:23:00.123730+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-BW
- **new_region_distribution:** [0.075, 0.075, 0.075, 0.7, 0.075]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 11437 |
| actual_num_traces | 2000 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1014 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-05-11T07:46:01.117245+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 300 |
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
| num_resources | 10 |
| num_roles | 3 |
| num_trace_variants | 22 |
| num_trace_variants_before_noise | 22 |
| num_traces | 2000 |
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
