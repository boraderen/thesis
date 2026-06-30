# Rheon log: res

## General parameters

| Parameter | Value |
| --- | --- |
| num_traces | 2000 |
| num_activities | 12 |
| num_resources | 7 |
| num_regions | 4 |
| tree_weights | sequence=0.6, choice=0.25, parallel=0.1, loop=0.05 |
| start_date | 2020-01-01T00:00:00 |
| end_date | 2020-03-24T03:30:18.230879 |
| inter_arrival | 60 |
| activity_duration | 30, 100 |
| waiting_time | 15, 50 |
| amount | 1000, 40000 |
| seed | 7 |
| generated_traces | 2242 |
| generated_events | 20405 |
| num_trace_variants | 185 |

## Base distributions

### Activities

| Activity | Dominant resource | Duration mean | Duration var | Waiting mean | Waiting var |
| --- | --- | --- | --- | --- | --- |
| a | res_01 | 39.53 | 100 | 14.34 | 50 |
| b | res_04 | 36.62 | 100 | 15.05 | 50 |
| c | res_07 | 23.4 | 100 | 15.64 | 50 |
| d | res_04 | 25.2 | 100 | 20.95 | 50 |
| e | res_06 | 38.97 | 100 | 18.51 | 50 |
| f | res_07 | 18.13 | 100 | 16.47 | 50 |
| g | res_06 | 37.71 | 100 | 20.87 | 50 |
| h | res_05 | 37.13 | 100 | 11.58 | 50 |
| i | res_04 | 29.23 | 100 | 10.92 | 50 |
| j | res_04 | 25.27 | 100 | 16.35 | 50 |
| k | res_02 | 24.68 | 100 | 9.53 | 50 |
| l | res_04 | 24.12 | 100 | 9.43 | 50 |

### Case attributes

- Amount: mean=1000, variance=40000
- Inter-arrival mean: 60 minutes
- Dominant region: region_2 (of region_1, region_2, region_3, region_4)
- Resources: res_01, res_02, res_03, res_04, res_05, res_06, res_07

## Drifts

### d01 — reassignment (resource, sudden)

- Drift point: 0.200 (2020-01-17T15:06:03.646176)
- New dominant resource per activity:
  | Activity | Resource (before → after) |
  | --- | --- |
  | a | res_01 → res_05 |
  | b | res_04 → res_03 |
  | c | res_07 → res_03 |
  | d | res_04 → res_03 |
  | e | res_06 → res_05 |
  | f | res_07 → res_02 |
  | g | res_06 → res_05 |
  | h | res_05 → res_02 |
  | i | res_04 → res_07 |
  | j | res_04 → res_07 |
  | k | res_02 → res_06 |
  | l | res_04 → res_06 |

### d02 — duration (resource, sudden)

- Drift point: 0.400 (2020-02-03T06:12:07.292352)
- Affected resources: res_01, res_02
- Processing time multiplied by 2.5

### d03 — workload (resource, sudden)

- Drift point: 0.800 (2020-03-07T12:24:14.584703)
- Workload factor: 1.6
- Traces added: 242, removed: 0
