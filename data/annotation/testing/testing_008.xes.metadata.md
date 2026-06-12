# Rheon log: `testing_008`

- **Events:** 4,800
- **Cases:** 800
- **Time horizon:** 2020-01-01T00:00:00+00:00 → 2020-02-08T08:11:08.388843+00:00
- **Injected drifts:** 1

## Injected drifts

### #1 · sudden service_time (resource)

- **Drift id:** `d01`
- **Change point:** 2020-01-18T16:57:09.275934+00:00
- **Affected columns:** org:resource, org:role
- **multiplier_changes:** {'res_001': {'old_multiplier': 1.0, 'new_multiplier': 1.4996}, 'res_002': {'old_multiplier': 1.0, 'new_multiplier': 2.4335}, 'res_003': {'old_multiplier': 1.0, 'new_multiplier': 1.1843}, 'res_004': {'old_multiplier': 1.0, 'new_multiplier': 1.0837}, 'res_005': {'old_multiplier': 1.0, 'new_multiplier': 1.8192}, 'res_006': {'old_multiplier': 1.0, 'new_multiplier': 2.3537}, 'res_007': {'old_multiplier': 1.0, 'new_multiplier': 2.6485}, 'res_008': {'old_multiplier': 1.0, 'new_multiplier': 2.2754}}
- **note:** multiplier > 1.0 means slower (longer wait + processing), < 1.0 means faster
- **resources:** all

## Configuration

| Setting | Value |
| --- | --- |
| actual_num_events | 4800 |
| actual_num_traces | 800 |
| avg_trace_length | 6 |
| choice_weight | 0.1 |
| duplicate_activity_prob | 0.0 |
| global_seed | 1037 |
| gradual_overlap_fraction | 0.1 |
| horizon_end | 2020-02-08T08:11:08.388843+00:00 |
| horizon_max_days | 365 |
| horizon_min_days | 180 |
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
| num_roles | 3 |
| num_trace_variants | 1 |
| num_trace_variants_before_noise | 1 |
| num_traces | 800 |
| or_weight | 0.01 |
| parallel_weight | 0.02 |
| recurring_period_fraction | 0.2 |
| regions | DE-NRW, DE-BY, DE-HE, DE-BW, DE-BE |
| sequence_weight | 0.85 |
| service_time_mean_min | 10 |
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
