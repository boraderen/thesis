"""Domain knowledge injected into prompts: what the method is and how to read it.

Four hardcoded knowledge blocks (drift detection, process mining, the kairo
method, statistics) plus one derived from the current log. `inject` composes
the requested parts into a single string for prompt assembly.
"""
from __future__ import annotations

import pandas as pd

from .abstract import abstract_config, abstract_log_attributes, abstract_log_statistics
from ..data.log import log_statistics

DRIFT_DETECTION = """\
The analysis studies state-based concept drift detection in traditional event logs.
A process is monitored through three complementary notions of state:
- Intra-case state: the situation of one running case, derived from its event prefix
  (executed activities, their order, position in the case).
- Resource state: how the resource dimension behaves across all cases in a calendar
  window (workload, handovers, waiting times, activity-resource assignments).
- Inter-case state: the global process situation per calendar window (active cases,
  arrivals, completions, pacing, stalled cases, case-attribute mixes).
Concept drift means the underlying process changed over time. The core idea: compute
states by clustering feature vectors, follow how state occupancy evolves over calendar
windows, and read significant changes in those state signals as drift indicators.
A good analysis names WHERE the signal spikes (which windows), WHAT changed (which
states grew or shrank, which features moved), and HOW confident the evidence is."""

PROCESS_MINING = """\
Process mining background:
- An event log records events; each event belongs to exactly one case and carries at
  least a case id, an activity name, and a timestamp; often also a resource.
- A trace is a case's events ordered by time; a variant is the activity sequence of a
  trace; a prefix is the first k events of a trace.
- Directly-follows (A→B): activity B occurred immediately after A within the same case.
- A handover is a within-case pair of consecutive events executed by different resources.
- Throughput time (TPT) is the duration from a case's first to its last event.
- Concept drift types: sudden (the process switches at a point), gradual (old and new
  coexist for a while), recurring (seasonal switching back and forth), incremental
  (many small steps). A single spike in a drift signal suggests sudden drift; a sustained
  elevated band suggests gradual or incremental drift; periodic spikes suggest recurring
  drift or seasonality."""

METHOD = """\
The kairo pipeline (identical shape for all three perspectives):
1. Feature extraction — intra-case: one vector per event describing its case prefix;
   resource and inter-case: one vector per calendar window of W minutes.
2. Optional PCA compresses the vectors; the elbow rule picks the number of components
   unless it is fixed. Optional standardization z-scores what goes into clustering.
3. Clustering turns vectors into discrete states: SOM (grid of cells, each a state),
   k-means (k states), or DBSCAN (density clusters plus a Noise state). The distance
   metric (euclidean, manhattan, chebyshev, cosine) shapes what counts as similar.
4. Each sample gets a state -> state trajectories over time; transitions are the points
   where the state changes, annotated with the features that moved most.
5. Drift signals: intra-case aggregates per-window state distributions and scores each
   window against a reference (previous window, mean of the last l windows, or the
   full-log baseline) with a divergence (KL, Jensen-Shannon, total variation,
   Hellinger). Resource and inter-case compare each window's compressed vector with
   the previous window's under a vector distance.
Parameter effects worth reasoning about: a larger SOM grid or k splits behaviour into
finer states (smaller ones merge it); DBSCAN's eps/min_samples decide how much lands
in Noise; the window size W trades temporal resolution against noise; the reference
choice decides whether a shift appears once (previous) or smears (baseline)."""

STATISTICS = """\
How to read the diagnostics:
- Explained variance (PCA): the share of the data's spread each component keeps; the
  elbow is where adding a component stops paying. Low total explained variance means
  the compressed space lost much of the structure.
- k-distance curve (DBSCAN): sorted distance of every point to its k-th neighbour; the
  knee is the usual eps candidate — points beyond it end up as noise.
- Silhouette: -1..1, how much closer samples sit to their own state than to the next
  one; near 0 means overlapping states.
- Divergences on distributions: KL is asymmetric and unbounded; Jensen-Shannon is its
  symmetric, bounded smoothing; total variation is half the L1 difference; Hellinger
  is bounded in [0,1] and less spike-prone than KL.
- A drift signal is evidence, not proof: check whether a spike aligns with a state's
  occupancy changing in the distribution, whether transitions cluster around it, and
  whether it persists across parameter choices before calling it drift."""

KNOWLEDGE = {
    "drift_detection": DRIFT_DETECTION,
    "process_mining": PROCESS_MINING,
    "method": METHOD,
    "statistics": STATISTICS,
}

DEFAULT_PARTS = ("drift_detection", "process_mining", "method", "statistics", "log")


def log_knowledge(log: pd.DataFrame, max_len: int = 2000) -> str:
    """What the current log looks like: statistics first, then column details."""
    stats_text = abstract_log_statistics(log_statistics(log), max_len=max_len // 2)
    columns_text = abstract_log_attributes(log, max_len=max_len // 2)
    return f"{stats_text}\n{columns_text}"


def inject(
    log: pd.DataFrame | None = None,
    config=None,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    max_len: int = 6000,
) -> str:
    """Compose the requested knowledge blocks into one string.

    `parts` may name any of {drift_detection, process_mining, method, statistics,
    log}; the log block needs `log`. A `config` appends the run's parameters.
    """
    blocks = [KNOWLEDGE[part] for part in parts if part in KNOWLEDGE]
    if log is not None and "log" in parts:
        blocks.append(log_knowledge(log, max_len=max(500, max_len // 3)))
    if config is not None:
        blocks.append(abstract_config(config))
    from .abstract import _clip

    return _clip("\n\n".join(blocks), max_len)
