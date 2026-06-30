# Rheon log: example

## General parameters

| Parameter | Value |
| --- | --- |
| num_traces | 2000 |
| num_activities | 8 |
| num_resources | 8 |
| num_regions | 4 |
| tree_weights | sequence=0.6, choice=0.25, parallel=0.1, loop=0.05 |
| start_date | 2020-01-01T00:00:00+00:00 |
| end_date | 2020-03-25T11:24:23.927705+00:00 |
| inter_arrival | 60 |
| activity_duration | 30, 100 |
| waiting_time | 15, 50 |
| amount | 1000, 40000 |
| seed | 42 |
| generated_traces | 2000 |
| generated_events | 10592 |
| num_trace_variants | 98 |

## Base distributions

### Activities

| Activity | Dominant resource | Duration mean | Duration var | Waiting mean | Waiting var |
| --- | --- | --- | --- | --- | --- |
| a | res_02 | 38.61 | 100 | 20.12 | 50 |
| b | res_07 | 34.74 | 100 | 16.73 | 50 |
| c | res_06 | 20.26 | 100 | 18.87 | 50 |
| d | res_03 | 41.41 | 100 | 14.32 | 50 |
| e | res_01 | 36.27 | 100 | 11.73 | 50 |
| f | res_08 | 36.87 | 100 | 15.66 | 50 |
| g | res_04 | 21.07 | 100 | 9.77 | 50 |
| h | res_08 | 28.81 | 100 | 18.93 | 50 |
| i | res_06 | 26.9 | 100 | 16.58 | 50 |

### Case attributes

- Amount: mean=1000, variance=40000
- Inter-arrival mean: 60 minutes
- Dominant region: region_4 (of region_1, region_2, region_3, region_4)
- Resources: res_01, res_02, res_03, res_04, res_05, res_06, res_07, res_08

## Drifts

### d01 — control_flow (intra-case, sudden)

- Drift point: 0.500 (2020-02-12T05:42:11.963853+00:00)
- New process tree: num_activities=9, weights sequence=0.6, choice=0.25, parallel=0.1, loop=0.05
