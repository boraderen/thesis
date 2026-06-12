# Rheon log: `training_014`

- **Events:** 4,822
- **Cases:** 834
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-02-28T09:36:00+00:00
- **Injected drifts:** 2

## Injected drifts

### #1 · sudden burstiness (inter_case)

- **Drift id:** `d01`
- **Change point:** 2020-01-25T06:46:46.291698+00:00
- **Affected columns:** case:concept:name, start_timestamp, time:timestamp
- **mean_preserved:** True
- **new_shape:** bimodal_pareto
- **old_shape:** exponential

### #2 · sudden tree_mutation (control_flow)

- **Drift id:** `d02`
- **Change point:** 2020-02-01T05:11:59.771876+00:00
- **Affected columns:** concept:name
- **activities_added:** ['b']
- **activities_deleted:** ['b']
- **activities_moved:** ['b', 'a', 'f', 'd', 'e', 'c', 'b', 'a', 'f', 'd', 'e', 'c']
- **operator_swaps:** [{'from': 'sequence', 'to': 'parallel', 'affected_activities': 'b,a,f,d,e,c'}, {'from': 'parallel', 'to': 'loop', 'affected_activities': 'b,a,f,d,e,c'}]
- **variant_count_after:** 2
- **variant_count_before:** 1
- **variant_counts_by_version:** [1, 2]

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 4822 |
| actual_num_traces | 834 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1013 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-02-28T09:36:00+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 365 |
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
| num_resources | 8 |
| num_roles | 2 |
| num_trace_variants | 3 |
| num_trace_variants_before_noise | 3 |
| num_traces | 800 |
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
