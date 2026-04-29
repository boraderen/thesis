import pm4py
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pathlib import Path
from datetime import datetime, timedelta

# ── Input log ─────────────────────────────────────────────────────────────────
LOG_PATH = "logs/sudden_drift.xes"
# ──────────────────────────────────────────────────────────────────────────────

# ── Cut timestamps (ISO 8601) ──────────────────────────────────────────────────
# Each timestamp is a boundary between two consecutive sublogs.
# N timestamps produce N+1 sublogs.
# A trace is assigned to the window in which its last event falls.
CUT_TIMESTAMPS = [
    "2021-06-21 09:22:04"
]
# ──────────────────────────────────────────────────────────────────────────────

log = pm4py.read_xes(LOG_PATH, return_legacy_log_object=True)

# Build window boundaries: sentinel start + cuts + sentinel end.
boundaries = ["1970-01-01 00:00:00"] + CUT_TIMESTAMPS + ["2099-12-31 00:00:00"]
windows = list(zip(boundaries[:-1], boundaries[1:]))

out_dir = Path(LOG_PATH).parent
stem    = Path(LOG_PATH).stem

print(f"Log:    {LOG_PATH}  ({len(log)} traces)")
print(f"Splits: {len(CUT_TIMESTAMPS)} cut points → {len(windows)} sublogs")
print()

for i, (t_start, t_end) in enumerate(windows):
    # pm4py uses closed [t_start, t_end] intervals, so subtract 1 s from the upper
    # bound to make it half-open [t_start, t_end): events at the cut go to the next window.
    t_end_exclusive = (
        datetime.fromisoformat(t_end) - timedelta(seconds=1)
    ).isoformat(sep=" ")
    sublog = pm4py.filter_time_range(
        log, t_start, t_end_exclusive,
        mode='events',  # event-level cut; traces spanning the boundary are split
    )
    out_path = out_dir / f"{stem}_window_{i}.xes"
    xes_exporter.apply(sublog, str(out_path))
    print(f"  window {i}  [{t_start}  →  {t_end_exclusive}]  {len(sublog):4d} traces  →  {out_path.name}")
