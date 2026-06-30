# Rheon log: intra

## General parameters


| Parameter          | Value                                              |
| ------------------ | -------------------------------------------------- |
| num_traces         | 2000                                               |
| num_activities     | 8                                                  |
| num_resources      | 8                                                  |
| num_regions        | 4                                                  |
| tree_weights       | sequence=0.6, choice=0.25, parallel=0.1, loop=0.05 |
| start_date         | 2020-01-01T00:00:00+00:00                          |
| end_date           | 2020-03-25T11:53:12.104008+00:00                   |
| inter_arrival      | 60                                                 |
| activity_duration  | 30, 100                                            |
| waiting_time       | 15, 50                                             |
| amount             | 1000, 40000                                        |
| seed               | 42                                                 |
| generated_traces   | 2000                                               |
| generated_events   | 9414                                               |
| num_trace_variants | 108                                                |


## Base distributions

### Activities


| Activity | Dominant resource | Duration mean | Duration var | Waiting mean | Waiting var |
| -------- | ----------------- | ------------- | ------------ | ------------ | ----------- |
| a        | res_04            | 34.74         | 100          | 18.87        | 50          |
| b        | res_08            | 20.26         | 100          | 14.32        | 50          |
| c        | res_06            | 41.41         | 100          | 11.73        | 50          |
| d        | res_07            | 36.27         | 100          | 15.66        | 50          |
| e        | res_07            | 36.87         | 100          | 9.77         | 50          |
| f        | res_02            | 21.07         | 100          | 18.93        | 50          |
| g        | res_03            | 28.81         | 100          | 16.58        | 50          |
| h        | res_04            | 26.9          | 100          | 18.1         | 50          |
| i        | res_04            | 40.24         | 100          | 13.25        | 50          |
| j        | res_01            | 33.45         | 100          | 20.65        | 50          |




### Case attributes

- Amount: mean=1000, variance=40000
- Inter-arrival mean: 60 minutes
- Dominant region: region_3 (of region_1, region_2, region_3, region_4)
- Resources: res_01, res_02, res_03, res_04, res_05, res_06, res_07, res_08



## Drifts



### d01 — control_flow (intra-case, sudden)

- Drift point: 0.400 (2020-02-03T19:09:16.841603+00:00)
- New process tree: num_activities=10, weights sequence=0.6, choice=0.25, parallel=0.1, loop=0.05



### d02 — control_flow (intra-case, gradual)

- Drift window: 0.700 → 0.850
(2020-02-29T03:31:14.472806+00:00 → 2020-03-12T19:42:13.288407+00:00)
- New process tree: num_activities=7, weights sequence=0.6, choice=0.25, parallel=0.1, loop=0.05

