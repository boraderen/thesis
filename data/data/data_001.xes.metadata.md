# Driftify log: `data_001`

- **Events:** 36,000
- **Cases:** 6,000
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2021-02-12T18:10:43.368542+00:00
- **Injected drifts:** 3

## Injected drifts

### #1 · sudden numeric (data)

- **Drift id:** `d01`
- **Change point:** 2020-04-28T10:13:06.657126+00:00
- **Affected columns:** case:amount
- **new_amount_mean:** 1450.0
- **new_amount_std:** 175.0
- **old_amount_mean:** 1000.0
- **old_amount_std:** 250.0

### #2 · sudden categorical (data)

- **Drift id:** `d02`
- **Change point:** 2020-08-03T15:32:31.004691+00:00
- **Affected columns:** case:region
- **dominant_region:** DE-HE
- **new_region_distribution:** [0.075, 0.075, 0.7, 0.075, 0.075]
- **old_region_distribution:** [0.2, 0.2, 0.2, 0.2, 0.2]

### #3 · sudden numeric (data)

- **Drift id:** `d03`
- **Change point:** 2020-10-22T08:03:26.559272+00:00
- **Affected columns:** case:amount
- **new_amount_mean:** 942.5
- **new_amount_std:** 280.0
- **old_amount_mean:** 1450.0
- **old_amount_std:** 175.0

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 36000 |
| actual_num_traces | 6000 |
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
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 6000 |
| or_weight | 0.01 |
| output_path | data/data-attributes/ |
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
