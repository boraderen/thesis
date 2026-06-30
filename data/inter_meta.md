# Rheon log: inter

## General parameters

| Parameter | Value |
| --- | --- |
| num_traces | 4000 |
| num_activities | 8 |
| num_resources | 8 |
| num_regions | 4 |
| tree_weights | sequence=0.6, choice=0.25, parallel=0.1, loop=0.05 |
| start_date | 2020-01-01T00:00:00+00:00 |
| end_date | 2020-12-31T00:00:00+00:00 |
| activity_duration | 30, 100 |
| waiting_time | 15, 50 |
| amount | 1000, 40000 |
| seed | 42 |
| generated_traces | 7304 |
| generated_events | 29330 |
| num_trace_variants | 4 |

## Base distributions

### Activities

| Activity | Dominant resource | Duration mean | Duration var | Waiting mean | Waiting var |
| --- | --- | --- | --- | --- | --- |
| a | res_08 | 28.53 | 100 | 14.4 | 50 |
| b | res_01 | 38.61 | 100 | 13.45 | 50 |
| c | res_07 | 34.74 | 100 | 20.12 | 50 |
| d | res_07 | 20.26 | 100 | 16.73 | 50 |
| e | res_03 | 41.41 | 100 | 18.87 | 50 |
| f | res_06 | 36.27 | 100 | 14.32 | 50 |
| g | res_02 | 36.87 | 100 | 11.73 | 50 |
| h | res_07 | 21.07 | 100 | 15.66 | 50 |

### Case attributes

- Amount: mean=1000, variance=40000
- Inter-arrival mean (derived from horizon / num_traces): 131.4 minutes
- Dominant region: region_3 (of region_1, region_2, region_3, region_4)
- Resources: res_01, res_02, res_03, res_04, res_05, res_06, res_07, res_08

## Drifts

### d01 — arrival_rate (inter-case, sudden)

- Drift point: 0.200 (2020-03-14T00:00:00+00:00)
- Inter-arrival mean: 131.4 → 65.7 minutes

### d02 — amount (inter-case, sudden)

- Drift point: 0.450 (2020-06-13T06:00:00+00:00)
- Amount mean: 1000 → 4000
- Amount variance: 40000 → 40000

### d03 — waiting_time (inter-case, sudden)

- Drift point: 0.650 (2020-08-25T06:00:00+00:00)
- Waiting-gap mean: 15.66 → 120
- Waiting-gap variance: 50 → 50

### d04 — region (inter-case, sudden)

- Drift point: 0.850 (2020-11-06T06:00:00+00:00)
- Dominant region: region_3 → region_1
