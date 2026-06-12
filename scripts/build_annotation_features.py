"""Extract tabular time-series features and labels from each annotation log.

For every XES file under `data/annotation/<split>/`, this script
  1. loads the log via the dashboard loader,
  2. builds inter-case, resource, and intra-case features (the same pipeline
     the dashboard pages use),
  3. trains a 3×3 SOM on the intra-case features to produce per-window state
     distributions,
  4. aggregates resource features to log-agnostic summaries,
  5. parses the matching `*.metadata.md` to locate change points and labels
     each window with binary drift / per-perspective targets,
  6. writes the merged table to `data/annotation/<split>/<stem>.features.csv`.

The output schema is identical across logs, so the notebook can simply
concat them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

from core.features.inter_case import build_features as build_inter
from core.features.intra_case import build_features as build_intra
from core.features.resource import build_features as build_resource
from core.loader import load_xes
from core.som import train_som
from core.windows import default_window_minutes, floor_to_window, log_span_minutes


PERSPECTIVES = ("control_flow", "data", "inter_case", "resource")
SOM_GRID = (3, 3)
SOM_STATES = SOM_GRID[0] * SOM_GRID[1]


def parse_metadata(path: Path) -> dict:
    """Pull injected drifts (perspective + change point) from the markdown sidecar."""
    text = path.read_text()
    drifts = []
    # Sections look like: "### #1 · sudden tree_mutation (control_flow)"
    sec_re = re.compile(r"###\s+#\d+\s+·\s+\w+\s+(\w+)\s+\((\w+)\)")
    cp_re = re.compile(r"\*\*Change point:\*\*\s+(\S+)")
    for match in sec_re.finditer(text):
        subtype, perspective = match.group(1), match.group(2)
        tail = text[match.end():]
        cp = cp_re.search(tail)
        if cp is None:
            continue
        drifts.append(
            dict(
                perspective=perspective,
                subtype=subtype,
                change_point=pd.Timestamp(cp.group(1)),
            )
        )
    return {"drifts": drifts}


def aggregate_resource(matrix: pd.DataFrame, spec) -> pd.DataFrame:
    """Reduce per-resource columns to log-agnostic summaries."""
    events_cols = [f"events:{r}" for r in spec.resources]
    active_cols = [f"active:{r}" for r in spec.resources]
    dur_cols = [f"duration:{r}" for r in spec.resources]
    wait_cols = [f"wait:{r}" for r in spec.resources]
    ho_cols = [c for c in matrix.columns if c.startswith("ho:")]

    events = matrix[events_cols].to_numpy() if events_cols else np.zeros((len(matrix), 1))
    active = matrix[active_cols].to_numpy() if active_cols else np.zeros((len(matrix), 1))
    dur = matrix[dur_cols].to_numpy() if dur_cols else np.zeros((len(matrix), 1))
    wait = matrix[wait_cols].to_numpy() if wait_cols else np.zeros((len(matrix), 1))
    ho = matrix[ho_cols].to_numpy() if ho_cols else np.zeros((len(matrix), 1))

    return pd.DataFrame(
        dict(
            res_events_total=events.sum(axis=1),
            res_events_max=events.max(axis=1),
            res_events_std=events.std(axis=1),
            res_active_total=active.sum(axis=1),
            res_n_busy=(events > 0).sum(axis=1).astype(float),
            res_duration_mean=dur.mean(axis=1),
            res_wait_mean=wait.mean(axis=1),
            res_handover_total=ho.sum(axis=1),
            res_handover_unique=(ho > 0).sum(axis=1).astype(float),
        )
    )


def intra_state_distribution(feat: pd.DataFrame, window_minutes: int, origin: pd.Timestamp) -> pd.DataFrame:
    """Per-window frequency vector over the 9 SOM states (matches dashboard logic)."""
    df = feat[["timestamp", "state_id"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["__win__"] = floor_to_window(df["timestamp"], origin, window_minutes)
    counts = (
        df.groupby(["__win__", "state_id"]).size().unstack(fill_value=0)
          .reindex(columns=range(SOM_STATES), fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    freqs = counts.div(totals, axis=0).fillna(0.0)
    freqs.columns = [f"intra_S{s}" for s in freqs.columns]
    freqs.index.name = "window_start"
    return freqs.reset_index().rename(columns={"__win__": "window_start"})


def label_windows(windows: pd.Series, window_minutes: int, drifts: list[dict]) -> pd.DataFrame:
    """Per-window multi-label drift targets (one column per perspective + 'any')."""
    width = pd.Timedelta(minutes=window_minutes)
    starts = pd.DatetimeIndex(windows)
    out = pd.DataFrame(
        {f"drift_{p}": np.zeros(len(starts), dtype=int) for p in PERSPECTIVES}
    )
    out["drift_any"] = 0
    for drift in drifts:
        cp = drift["change_point"]
        if cp.tzinfo is None:
            cp = cp.tz_localize("UTC")
        mask = (starts <= cp) & (cp < starts + width)
        if mask.any():
            out.loc[mask, f"drift_{drift['perspective']}"] = 1
            out.loc[mask, "drift_any"] = 1
    return out


def extract_one(xes_path: Path) -> pd.DataFrame:
    """Build the full per-window feature+label table for a single log."""
    log = load_xes(xes_path.read_bytes())
    window_minutes = default_window_minutes(log_span_minutes(log))
    origin = log["timestamp"].min()

    inter_df, _ = build_inter(log, window_minutes=window_minutes, stall_minutes=60)

    if "resource" in log.columns:
        res_df, res_spec = build_resource(log, window_minutes=window_minutes)
        res_agg = aggregate_resource(res_df, res_spec)
        res_agg.insert(0, "window_start", pd.DatetimeIndex(res_df["window_start"]).tz_convert("UTC"))
    else:
        res_agg = pd.DataFrame({"window_start": inter_df["window_start"]})

    intra_feat, _ = build_intra(log, window=3)
    som = train_som(
        intra_feat.drop(columns=["case_id", "activity", "timestamp"]).to_numpy(),
        grid_h=SOM_GRID[0], grid_w=SOM_GRID[1],
    )
    intra_feat = intra_feat.assign(state_id=som.state_ids)
    intra_dist = intra_state_distribution(intra_feat, window_minutes, origin)

    table = inter_df.merge(res_agg, on="window_start", how="left").merge(
        intra_dist, on="window_start", how="left"
    )
    table = table.fillna(0.0)

    meta = parse_metadata(xes_path.with_suffix(".xes.metadata.md"))
    labels = label_windows(table["window_start"], window_minutes, meta["drifts"])
    return pd.concat([table.reset_index(drop=True), labels], axis=1)


def process_split(split_dir: Path) -> None:
    for xes_path in sorted(split_dir.glob("*.xes")):
        out_path = xes_path.with_suffix(".features.csv")
        if out_path.exists():
            print(f"skip {out_path.name} (exists)")
            continue
        print(f"extract {xes_path.name}")
        table = extract_one(xes_path)
        table.insert(0, "log_id", xes_path.stem)
        table.to_csv(out_path, index=False)


if __name__ == "__main__":
    base = ROOT / "data" / "annotation"
    for split in ("training", "testing"):
        process_split(base / split)
    print("Done.")
