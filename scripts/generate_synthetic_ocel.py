"""Generate a synthetic OCEL 1.0 (.jsonocel) log for thesis experiments.

Design philosophy
-----------------
We generate cases (orders) one at a time. Each case is built by:

1. Sampling a *case template* (a "happy variant" or one of a handful of
   deviation variants). The variants are explicit lists of (activity,
   object_subset_selector) pairs. Variant probabilities are heavily skewed,
   which produces a Pareto-like variant mix without us having to
   post-process anything.

2. Sampling case-level attributes (city/country, item_count, order_value,
   shipment_company, shipping_service). These bias the per-step durations
   (e.g. high item_count -> longer Pack Items; Express shipping ->
   shorter delivery).

3. Walking the variant step by step, allocating a resource from the right
   pool (warehouse / courier / csr / system), and computing event start +
   complete timestamps using activity-specific duration distributions.

4. The case "creation time" (Place Order) is sampled from a non-uniform
   arrival process that combines yearly seasonality (summer slump, sharp
   Q4 peak), a weekday/weekend bias, and a bias toward evening hours.
   After each concept drift, the global arrival rate jumps so that
   events-per-time visibly changes.

Concept drifts (observable in process discovery / drift detection):
  - drift_1 (~33% through the period): quality-check skip rate climbs
    sharply. New deviation variant becomes more common.
  - drift_2 (~66% through the period): activity ordering changes
    (Quality Check moves to before Pack Items), express-shipping share
    doubles, and the global arrival rate increases by ~50%.

Output is OCEL 1.0 jsonocel because that is what the existing sample log
in logs/ uses. pm4py's classic jsonocel exporter promotes any non-
``ocel:``-prefixed column on the events DataFrame to the per-event vmap,
and the same for objects -> ovmap. We rely on that convention, so all
custom attributes (state, duration, lifecycle:transition,
start:timestamp, org:resource, city, country, ...) are stored as plain
columns alongside the canonical ``ocel:eid``/``ocel:activity``/
``ocel:timestamp`` columns.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import pm4py
from pm4py.objects.ocel.obj import OCEL


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

ACTIVITIES = [
    "Place Order",        # csr / online (system)
    "Confirm Order",      # csr
    "Pay Order",          # payment system
    "Pick Item",          # warehouse worker (per item)
    "Pack Items",         # warehouse worker (per package)
    "Quality Check",      # warehouse worker
    "Ship Package",       # courier (handoff)
    "In Transit",         # courier (logistics scan)
    "Deliver Package",    # courier (last mile)
    "Cancel Order",       # csr (only on the cancel deviation)
]

OBJECT_TYPES = ["order", "item", "package"]

# Resource pools. Sizes tuned so that pools stay busy but no single
# worker dominates. Resource IDs are stable across the whole log.
POOL_SIZES = {
    "warehouse": 18,   # Pick / Pack / Quality Check
    "courier": 12,     # Ship / In Transit / Deliver
    "csr": 6,          # Place / Confirm / Cancel
}
SYSTEM_RESOURCE = "payment-svc"  # Pay Order is automated

ACTIVITY_TO_POOL = {
    "Place Order": "csr",
    "Confirm Order": "csr",
    "Pay Order": "system",
    "Pick Item": "warehouse",
    "Pack Items": "warehouse",
    "Quality Check": "warehouse",
    "Ship Package": "courier",
    "In Transit": "courier",
    "Deliver Package": "courier",
    "Cancel Order": "csr",
}

# Per-event "state" attribute (lightweight FSM marker shown in dashboards).
ACTIVITY_STATE = {
    "Place Order": "placed",
    "Confirm Order": "confirmed",
    "Pay Order": "paid",
    "Pick Item": "picking",
    "Pack Items": "packed",
    "Quality Check": "qc_passed",
    "Ship Package": "shipped",
    "In Transit": "in_transit",
    "Deliver Package": "delivered",
    "Cancel Order": "cancelled",
}

CITY_COUNTRY = [
    ("Berlin", "DE"), ("Munich", "DE"), ("Hamburg", "DE"),
    ("Vienna", "AT"), ("Zurich", "CH"),
    ("Paris", "FR"), ("Lyon", "FR"),
    ("Amsterdam", "NL"), ("Madrid", "ES"), ("Milan", "IT"),
]
SHIPMENT_COMPANIES = ["DHL", "UPS", "FedEx", "Hermes"]
SHIPPING_SERVICES = ["Standard", "Express"]


# ---------------------------------------------------------------------------
# Variants ("happy" + deviations)
# ---------------------------------------------------------------------------
#
# A variant step is (activity, object_selector). The selector tells us
# which objects to attach to the event:
#   - "order"          -> the order object only
#   - "all_items"      -> every item of the order (one event total)
#   - "per_item"       -> emit one event per item (item + order)
#   - "package"        -> the package object only
#   - "package+order"  -> package + order
#
# Variant *weights* are intentionally skewed so the empirical distribution
# is Pareto-like: one dominant happy path, then a steep drop-off.

HAPPY_PRE_DRIFT = [
    ("Place Order",      "order_and_items"),
    ("Confirm Order",    "order_and_items"),
    ("Pay Order",        "order"),
    ("Pick Item",        "per_item"),
    ("Pack Items",       "package_order_items"),
    ("Quality Check",    "package"),
    ("Ship Package",     "package_order"),
    ("In Transit",       "package"),
    ("Deliver Package",  "package_order"),
]

# After drift_2, Quality Check is moved to *before* Pack Items.
HAPPY_POST_DRIFT2 = [
    ("Place Order",      "order_and_items"),
    ("Confirm Order",    "order_and_items"),
    ("Pay Order",        "order"),
    ("Pick Item",        "per_item"),
    ("Quality Check",    "package_items"),   # checked items, not the package
    ("Pack Items",       "package_order_items"),
    ("Ship Package",     "package_order"),
    ("In Transit",       "package"),
    ("Deliver Package",  "package_order"),
]


def variant_skip_qc(base):
    return [step for step in base if step[0] != "Quality Check"]


def variant_repick(base):
    out = []
    for step in base:
        out.append(step)
        if step[0] == "Pick Item":
            out.append(step)  # one extra Pick Item pass
    return out


def variant_cancel_after_confirm(_base):
    return [
        ("Place Order",   "order_and_items"),
        ("Confirm Order", "order_and_items"),
        ("Cancel Order",  "order"),
    ]


def variant_cancel_after_pay(_base):
    return [
        ("Place Order",   "order_and_items"),
        ("Confirm Order", "order_and_items"),
        ("Pay Order",     "order"),
        ("Cancel Order",  "order"),
    ]


def variant_qc_twice(base):
    out = []
    for step in base:
        out.append(step)
        if step[0] == "Quality Check":
            out.append(step)
    return out


def build_variant_pool(base_happy):
    """Returns list of (variant_steps, weight) tuples.

    Weights are unnormalised; the caller normalises. The weights are the
    knob that controls the Pareto-like variant mix.
    """
    return [
        (base_happy,                          70.0),  # happy path
        (variant_skip_qc(base_happy),          8.0),
        (variant_repick(base_happy),           6.0),
        (variant_qc_twice(base_happy),         3.0),
        (variant_cancel_after_pay(base_happy), 2.0),
        (variant_cancel_after_confirm(base_happy), 1.5),
    ]


# ---------------------------------------------------------------------------
# Duration distributions (in seconds)
# ---------------------------------------------------------------------------

def duration_for(activity, *, item_count, shipping_service, rng):
    """Return (mean_seconds, sigma_seconds) for an activity, with the
    correlations the user asked for baked in."""
    if activity == "Place Order":
        return rng.lognormal(mean=math.log(60), sigma=0.4)
    if activity == "Confirm Order":
        return rng.lognormal(mean=math.log(45), sigma=0.5)
    if activity == "Pay Order":
        return rng.lognormal(mean=math.log(20), sigma=0.6)
    if activity == "Pick Item":
        return rng.lognormal(mean=math.log(120), sigma=0.5)
    if activity == "Pack Items":
        # Strong correlation: more items -> longer packing.
        base = 60 + 25 * item_count
        return max(20.0, rng.normal(base, base * 0.2))
    if activity == "Quality Check":
        return rng.lognormal(mean=math.log(90 + 10 * item_count), sigma=0.3)
    if activity == "Ship Package":
        return rng.lognormal(mean=math.log(300), sigma=0.4)
    if activity == "In Transit":
        # Most of the time-to-deliver is here; express is much faster.
        if shipping_service == "Express":
            mean = 14 * 3600   # ~14 h
        else:
            mean = 3 * 24 * 3600  # ~3 days
        return max(3600.0, rng.lognormal(mean=math.log(mean), sigma=0.4))
    if activity == "Deliver Package":
        # Last-mile leg; express still wins but the gap is smaller here.
        if shipping_service == "Express":
            mean = 30 * 60
        else:
            mean = 90 * 60
        return max(300.0, rng.lognormal(mean=math.log(mean), sigma=0.5))
    if activity == "Cancel Order":
        return rng.lognormal(mean=math.log(120), sigma=0.6)
    return 60.0


def gap_after(activity, *, rng):
    """Idle gap between successive events of the same case (seconds).

    Models real-world waits (queueing, customer reaction, network delay).
    """
    if activity in ("Place Order", "Confirm Order"):
        return rng.exponential(120)
    if activity == "Pay Order":
        return rng.exponential(600)         # customer takes time to pay
    if activity in ("Pick Item", "Pack Items", "Quality Check"):
        return rng.exponential(180)
    if activity == "Ship Package":
        return rng.exponential(1800)
    if activity == "In Transit":
        return rng.exponential(60)
    return rng.exponential(60)


# ---------------------------------------------------------------------------
# Arrival process: yearly seasonality + weekly + intraday + drift
# ---------------------------------------------------------------------------

def seasonal_multiplier(ts: datetime) -> float:
    """Yearly seasonality: summer slump (Jul-Aug), sharp Q4 peak (Nov-mid Dec)."""
    doy = ts.timetuple().tm_yday
    base = 1.0
    # Q4 sharp peak centred on day 330 (~Nov 26), fairly narrow.
    q4_peak = 1.8 * math.exp(-((doy - 330) ** 2) / (2 * 18 ** 2))
    # Summer slump centred on day 205 (~Jul 24), wider trough.
    summer_slump = -0.55 * math.exp(-((doy - 205) ** 2) / (2 * 30 ** 2))
    return max(0.15, base + q4_peak + summer_slump)


def weekly_multiplier(ts: datetime) -> float:
    # Weekends *higher* for order creation (people shop in their free time).
    return {0: 0.9, 1: 0.9, 2: 0.95, 3: 1.0, 4: 1.1, 5: 1.45, 6: 1.4}[ts.weekday()]


def intraday_multiplier(ts: datetime) -> float:
    """Bias toward evening hours (18-23) for order creation."""
    h = ts.hour + ts.minute / 60.0
    # Two soft humps: a small lunchtime bump and a big evening bump.
    lunch = 0.4 * math.exp(-((h - 13) ** 2) / (2 * 1.5 ** 2))
    evening = 1.3 * math.exp(-((h - 21) ** 2) / (2 * 2.0 ** 2))
    night_floor = 0.05 if (h < 6 or h > 23.5) else 0.25
    return night_floor + lunch + evening


def arrival_intensity(ts: datetime, base_rate: float, drift_factor: float) -> float:
    """Events per hour at time ts (Place Order arrivals)."""
    return base_rate * drift_factor * seasonal_multiplier(ts) * weekly_multiplier(ts) * intraday_multiplier(ts)


def sample_arrivals(start: datetime, end: datetime, target_orders: int,
                    drift_2_at: datetime, post_drift2_throughput: float,
                    rng: np.random.Generator) -> list[datetime]:
    """Thinning algorithm for a non-homogeneous Poisson process.

    We tune ``base_rate`` so we hit roughly ``target_orders`` arrivals.
    """
    # Estimate average multiplier by Monte Carlo over a calendar grid so
    # we can pick base_rate.
    grid = []
    cursor = start
    step = timedelta(hours=6)
    while cursor < end:
        m = seasonal_multiplier(cursor) * weekly_multiplier(cursor) * intraday_multiplier(cursor)
        if cursor >= drift_2_at:
            m *= post_drift2_throughput
        grid.append(m)
        cursor += step
    avg_mult = float(np.mean(grid)) if grid else 1.0

    total_hours = (end - start).total_seconds() / 3600.0
    # base_rate is orders per hour at multiplier == 1.
    base_rate = target_orders / max(1e-6, total_hours * avg_mult)

    # Rough upper bound on intensity for thinning.
    lam_max = base_rate * post_drift2_throughput * 3.5 * 1.5 * 1.7  # season * week * intraday peaks

    arrivals: list[datetime] = []
    t = start
    while t < end:
        # Inter-arrival under the dominating Poisson(lam_max), in hours.
        u = rng.random()
        dt_hours = -math.log(max(u, 1e-12)) / lam_max
        t = t + timedelta(hours=dt_hours)
        if t >= end:
            break
        drift_factor = post_drift2_throughput if t >= drift_2_at else 1.0
        intensity = arrival_intensity(t, base_rate, drift_factor)
        if rng.random() < intensity / lam_max:
            arrivals.append(t)
    return arrivals


# ---------------------------------------------------------------------------
# Resource pool with per-resource "next free" tracking
# ---------------------------------------------------------------------------

@dataclass
class ResourcePool:
    name: str
    members: list[str]
    next_free: dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self):
        for m in self.members:
            self.next_free.setdefault(m, datetime.min)

    def acquire(self, earliest_start: datetime, rng: random.Random) -> tuple[str, datetime]:
        """Pick a resource that is free closest to ``earliest_start``.

        We sample with a small random tie-breaker so workload spreads
        instead of always going to member[0].
        """
        # Pick the 3 resources with the earliest "next free" time, then
        # randomise among them to spread load.
        sorted_members = sorted(self.members, key=lambda m: self.next_free[m])
        candidates = sorted_members[: min(3, len(sorted_members))]
        chosen = rng.choice(candidates)
        start = max(earliest_start, self.next_free[chosen])
        return chosen, start

    def release(self, member: str, end_time: datetime) -> None:
        self.next_free[member] = end_time


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

@dataclass
class CaseContext:
    order_id: str
    item_ids: list[str]
    package_id: str
    item_count: int
    order_value: float
    city: str
    country: str
    shipment_company: str
    shipping_service: str


def select_objects(selector: str, ctx: CaseContext) -> list[tuple[str, str]]:
    """Resolve a variant step's selector into (object_id, object_type) pairs."""
    if selector == "order":
        return [(ctx.order_id, "order")]
    if selector == "package":
        return [(ctx.package_id, "package")]
    if selector == "package_order":
        return [(ctx.package_id, "package"), (ctx.order_id, "order")]
    if selector == "package_items":
        return [(ctx.package_id, "package")] + [(i, "item") for i in ctx.item_ids]
    if selector == "package_order_items":
        return ([(ctx.package_id, "package"), (ctx.order_id, "order")]
                + [(i, "item") for i in ctx.item_ids])
    if selector == "order_and_items":
        return [(ctx.order_id, "order")] + [(i, "item") for i in ctx.item_ids]
    raise ValueError(f"unknown selector {selector!r}")


def generate_case_attributes(rng_np: np.random.Generator, drift_express_boost: float) -> dict:
    item_count = int(np.clip(rng_np.geometric(p=0.35), 1, 12))
    avg_item_value = float(np.clip(rng_np.lognormal(mean=math.log(28), sigma=0.6), 3, 400))
    order_value = round(item_count * avg_item_value, 2)
    city, country = CITY_COUNTRY[int(rng_np.integers(0, len(CITY_COUNTRY)))]
    shipment_company = SHIPMENT_COMPANIES[int(rng_np.integers(0, len(SHIPMENT_COMPANIES)))]
    p_express = min(0.85, 0.18 * drift_express_boost)
    shipping_service = "Express" if rng_np.random() < p_express else "Standard"
    return dict(
        item_count=item_count, order_value=order_value, city=city, country=country,
        shipment_company=shipment_company, shipping_service=shipping_service,
    )


def pick_variant(variants: list[tuple[list, float]], rng: random.Random) -> list:
    weights = [w for _, w in variants]
    chosen = rng.choices(variants, weights=weights, k=1)[0]
    return list(chosen[0])


def generate_log(num_events: int, *, seed: int = 7,
                 start_date: datetime = datetime(2023, 1, 1),
                 end_date: datetime = datetime(2025, 12, 31)) -> OCEL:
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    # Drift schedule (absolute timestamps inside the period).
    span = end_date - start_date
    drift_1_at = start_date + span * (1 / 3)
    drift_2_at = start_date + span * (2 / 3)

    # Average events per case under the happy variant (Place + Confirm + Pay
    # + per-item Pick + Pack + QC + Ship + Transit + Deliver
    # = 8 + average_item_count). Use 9 as a working estimate.
    avg_events_per_case = 9.0
    target_orders = max(50, int(round(num_events / avg_events_per_case)))

    arrivals = sample_arrivals(
        start_date, end_date, target_orders,
        drift_2_at=drift_2_at, post_drift2_throughput=1.5, rng=rng_np,
    )

    pools = {
        "warehouse": ResourcePool("warehouse", [f"wh-{i:02d}" for i in range(POOL_SIZES["warehouse"])]),
        "courier":   ResourcePool("courier",   [f"cr-{i:02d}" for i in range(POOL_SIZES["courier"])]),
        "csr":       ResourcePool("csr",       [f"csr-{i:02d}" for i in range(POOL_SIZES["csr"])]),
    }

    events_rows: list[dict] = []
    relations_rows: list[dict] = []
    objects_rows: list[dict] = []
    seen_object_ids: set[str] = set()

    next_event_idx = 0
    next_item_id = 0
    next_package_id = 0

    for case_idx, place_ts in enumerate(arrivals):
        # Drift-aware case attributes / variant pool / pre-event pre-processing.
        if place_ts >= drift_2_at:
            base_happy = HAPPY_POST_DRIFT2
            express_boost = 2.5
            qc_skip_boost = 1.0  # drift_1's effect persists / overshadowed
        elif place_ts >= drift_1_at:
            base_happy = HAPPY_PRE_DRIFT
            express_boost = 1.0
            qc_skip_boost = 5.0   # quality team understaffed -> skip QC more
        else:
            base_happy = HAPPY_PRE_DRIFT
            express_boost = 1.0
            qc_skip_boost = 1.0

        variants = build_variant_pool(base_happy)
        # Apply drift_1 by up-weighting variants that have NO Quality Check.
        if qc_skip_boost > 1.0:
            variants = [
                (steps, w * (qc_skip_boost
                             if not any(s[0] == "Quality Check" for s in steps)
                             else 1.0))
                for steps, w in variants
            ]
        steps = pick_variant(variants, rng_py)

        attrs = generate_case_attributes(rng_np, express_boost)

        order_id = f"order_{case_idx:06d}"
        item_ids = [f"item_{next_item_id + k:07d}" for k in range(attrs["item_count"])]
        next_item_id += attrs["item_count"]
        package_id = f"pkg_{next_package_id:06d}"
        next_package_id += 1

        ctx = CaseContext(
            order_id=order_id, item_ids=item_ids, package_id=package_id,
            **attrs,
        )

        # Register objects (first occurrence only).
        for oid, otype in [(order_id, "order"), (package_id, "package")] + [(i, "item") for i in item_ids]:
            if oid in seen_object_ids:
                continue
            seen_object_ids.add(oid)
            row = {"ocel:oid": oid, "ocel:type": otype}
            if otype == "order":
                row.update({
                    "city": ctx.city,
                    "country": ctx.country,
                    "item_count": ctx.item_count,
                    "order_value": ctx.order_value,
                })
            elif otype == "package":
                row.update({
                    "shipment_company": ctx.shipment_company,
                    "shipping_service": ctx.shipping_service,
                })
            objects_rows.append(row)

        # Walk the variant. We expand "per_item" Pick Item into one event per item.
        cursor_ts = place_ts
        for activity, selector in steps:
            if activity == "Pick Item" and selector == "per_item":
                expanded = [(activity, ("single_item", iid)) for iid in item_ids]
            else:
                expanded = [(activity, selector)]

            for act, sel in expanded:
                # Resolve attached objects.
                if isinstance(sel, tuple) and sel[0] == "single_item":
                    iid = sel[1]
                    attached = [(iid, "item"), (order_id, "order")]
                else:
                    attached = select_objects(sel, ctx)

                # Acquire resource from the activity's pool.
                pool_name = ACTIVITY_TO_POOL[act]
                if pool_name == "system":
                    resource = SYSTEM_RESOURCE
                    start_ts = cursor_ts
                else:
                    resource, start_ts = pools[pool_name].acquire(cursor_ts, rng_py)

                dur_seconds = float(duration_for(
                    act, item_count=ctx.item_count,
                    shipping_service=ctx.shipping_service, rng=rng_np,
                ))
                end_ts = start_ts + timedelta(seconds=dur_seconds)
                if pool_name != "system":
                    pools[pool_name].release(resource, end_ts)

                eid = f"e_{next_event_idx:08d}"
                next_event_idx += 1

                event_row = {
                    "ocel:eid": eid,
                    "ocel:activity": act,
                    # ocel:timestamp is the canonical complete-timestamp; pm4py
                    # adds time:timestamp automatically when flattening, so we
                    # do NOT also store time:timestamp as a vmap entry (would
                    # produce duplicate columns post-flatten).
                    "ocel:timestamp": end_ts,
                    "start:timestamp": start_ts,
                    "lifecycle:transition": "complete",
                    "state": ACTIVITY_STATE[act],
                    "duration": round(dur_seconds, 3),
                    "city": ctx.city,
                    "country": ctx.country,
                    "item_count": ctx.item_count,
                    "order_value": ctx.order_value,
                    "org:resource": resource,
                    "shipment_company": ctx.shipment_company,
                    "shipping_service": ctx.shipping_service,
                }
                events_rows.append(event_row)

                for oid, otype in attached:
                    relations_rows.append({
                        "ocel:eid": eid,
                        "ocel:activity": act,
                        "ocel:timestamp": end_ts,
                        "ocel:oid": oid,
                        "ocel:type": otype,
                        "ocel:qualifier": "",
                    })

                # Advance the case cursor: completion + small idle gap.
                cursor_ts = end_ts + timedelta(seconds=float(gap_after(act, rng=rng_np)))

            if len(events_rows) >= num_events:
                break
        if len(events_rows) >= num_events:
            break

    events_df = pd.DataFrame(events_rows).sort_values("ocel:timestamp").reset_index(drop=True)
    relations_df = pd.DataFrame(relations_rows).sort_values("ocel:timestamp").reset_index(drop=True)
    objects_df = pd.DataFrame(objects_rows)

    ocel = OCEL()
    ocel.events = events_df
    ocel.objects = objects_df
    ocel.relations = relations_df
    return ocel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate a synthetic OCEL log.")
    p.add_argument("--num-events", type=int, default=20_000,
                   help="Approximate number of events to generate (hyperparameter).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start-date", type=str, default="2023-01-01")
    p.add_argument("--end-date",   type=str, default="2025-12-31")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "logs" / "synthetic_ocel.jsonocel"))
    return p.parse_args()


def main():
    args = parse_args()
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)

    ocel = generate_log(
        num_events=args.num_events, seed=args.seed,
        start_date=start_date, end_date=end_date,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_ocel(ocel, str(out_path))

    print(json.dumps({
        "out": str(out_path),
        "events": int(len(ocel.events)),
        "objects": int(len(ocel.objects)),
        "relations": int(len(ocel.relations)),
        "object_types": sorted(ocel.objects["ocel:type"].unique().tolist()),
        "activities": sorted(ocel.events["ocel:activity"].unique().tolist()),
        "first_ts": str(ocel.events["ocel:timestamp"].min()),
        "last_ts": str(ocel.events["ocel:timestamp"].max()),
    }, indent=2))


if __name__ == "__main__":
    main()
