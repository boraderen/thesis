# Rheon log: `training_002`

- **Events:** 12,000
- **Cases:** 2,000
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-05-06T15:06:50.091422+00:00
- **Injected drifts:** 1

## Injected drifts

### #1 · sudden numeric (data)

- **Drift id:** `d01`
- **Change point:** 2020-02-21T01:11:19.203923+00:00
- **Affected columns:** case:amount
- **new_amount_mean:** 650.0
- **new_amount_std:** 175.0
- **old_amount_mean:** 1000.0
- **old_amount_std:** 250.0

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 12000 |
| actual_num_traces | 2000 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1001 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-05-06T15:06:50.091422+00:00 |
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
| num_resources | 10 |
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 2000 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 20 |
| service_time_std_min | 7 |
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
