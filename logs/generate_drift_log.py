from conceptdrift.drifts.sudden import generate_log_with_sudden_drift
from conceptdrift.drifts.gradual import generate_log_with_gradual_drift
from conceptdrift.drifts.recurring import generate_log_with_recurring_drift
from conceptdrift.drifts.incremental import generate_log_with_incremental_drift
from pm4py.algo.simulation.tree_generator import algorithm as tree_gen
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

# ── Toggle which drift types to generate ──────────────────────────────────────
GENERATE_SUDDEN      = True
GENERATE_GRADUAL     = False
GENERATE_RECURRING   = False
GENERATE_INCREMENTAL = False
# ──────────────────────────────────────────────────────────────────────────────

# ── Trace-length tuning (PM4Py tree generator) ────────────────────────────────
# PM4Py uses mode/min/max as a triangular distribution for the number of
# visible activities in the generated process tree. Raising these values is the
# safest way to get longer traces without immediately exploding the number of
# trace variants.
VISIBLE_ACTIVITY_MODE = 24  # Most likely number of visible activities.
VISIBLE_ACTIVITY_MIN = 18   # Lower bound for visible activities.
VISIBLE_ACTIVITY_MAX = 35   # Upper bound for visible activities.

# PM4Py treats the operator values below as relative weights when it chooses
# which operator to insert next. They do not need to sum to 1.
#
# Longer traces:
# - more sequence: both branches execute in order, so traces stay predictably long
# - less choice: XOR executes only one branch, so high values tend to shorten traces
# - a little loop/or: can lengthen traces, but too much raises variability quickly
SEQUENCE_WEIGHT = 0.70
CHOICE_WEIGHT = 0.20
LOOP_WEIGHT = 0.02
OR_WEIGHT = 0.01

# Variant-control knobs:
# - parallel is the fastest route to combinatorial trace-variant growth
# - loop also increases variants, so keep it modest unless you explicitly want
#   more repetition and a wider trace-length spread
# - silent adds skip behavior, which often shortens visible traces
PARALLEL_WEIGHT = 0.000001
SILENT_WEIGHT = 0.0
DUPLICATE_WEIGHT = 0.0
NUMBER_OF_MODELS = 1  # cdlg expects one ProcessTree, not a list of trees

TREE_PARAMS = {
    'mode': VISIBLE_ACTIVITY_MODE,
    'min': VISIBLE_ACTIVITY_MIN,
    'max': VISIBLE_ACTIVITY_MAX,
    'sequence': SEQUENCE_WEIGHT,
    'choice': CHOICE_WEIGHT,
    'parallel': PARALLEL_WEIGHT,
    'loop': LOOP_WEIGHT,
    'or': OR_WEIGHT,
    'silent': SILENT_WEIGHT,
    'duplicate': DUPLICATE_WEIGHT,
    'no_models': NUMBER_OF_MODELS,
}
# ──────────────────────────────────────────────────────────────────────────────

# ── Case id normalization ─────────────────────────────────────────────────────
# CDLG/PM4Py generate trace names as plain strings by default. Zero-padding keeps
# them readable while also making lexicographic sorting match numeric order.
CASE_ID_START = 1
CASE_ID_KEY = 'concept:name'
CASE_ID_MIN_WIDTH = 4  # Small logs start at 0001; larger logs widen as needed.
# ──────────────────────────────────────────────────────────────────────────────


def make_tree():
    return tree_gen.apply(parameters=TREE_PARAMS)


def assign_padded_case_ids(log, start=CASE_ID_START, min_width=CASE_ID_MIN_WIDTH):
    """Assign consecutive zero-padded string case ids to trace concept:name."""
    last_case_id = start + len(log) - 1
    width = max(min_width, len(str(last_case_id)))
    for offset, trace in enumerate(log):
        trace.attributes[CASE_ID_KEY] = f"{start + offset:0{width}d}"
    return log


def export_log(log, output_path):
    assign_padded_case_ids(log)
    xes_exporter.apply(log, output_path)


if GENERATE_SUDDEN:
    # Abrupt switch from v1 to v2 at a single change point.
    log = generate_log_with_sudden_drift(
        num_traces        = 10000,      # total traces in the log
        change_point      = 0.5,       # drift occurs halfway through (trace 500)
        model_one         = make_tree(),
        model_two         = None,      # auto-evolve v2 from v1
        change_proportion = 0.4,       # ~30% of activities differ between v1 and v2
    )
    export_log(log, "logs/sudden_drift2.xes")
    print("=== sudden ===")
    print(log.attributes['drift info'])
    print()

if GENERATE_GRADUAL:
    # v1 and v2 traces are interleaved over a transition window; v2 becomes increasingly likely.
    log = generate_log_with_gradual_drift(
        num_traces        = 1000,      # total traces in the log
        start_point       = 0.4,       # transition window opens at trace 400
        end_point         = 0.6,       # transition window closes at trace 600
        distribution_type = 'linear',  # 'linear' or 'exponential' mix of v1/v2
        process_tree_one  = make_tree(),
        process_tree_two  = None,      # auto-evolve v2 from v1
        change_proportion = 0.3,
    )
    export_log(log, "logs/gradual_drift.xes")
    print("=== gradual ===")
    print(log.attributes['drift info'])
    print()

if GENERATE_RECURRING:
    # v1 and v2 alternate seasonally within the defined window.
    log = generate_log_with_recurring_drift(
        num_traces              = 1000,  # total traces in the log
        start_point             = 0.2,   # recurring pattern starts at trace 200
        end_point               = 0.8,   # recurring pattern ends at trace 800
        num_of_seasonal_changes = 3,     # how many times the process alternates v1↔v2
        pro_first_version       = 0.5,   # proportion of time spent in v1 within each cycle
        model_one               = make_tree(),
        model_two               = None,  # auto-evolve v2 from v1
        change_proportion       = 0.3,
    )
    export_log(log, "logs/recurring_drift.xes")
    print("=== recurring ===")
    print(log.attributes['drift info'])
    print()

if GENERATE_INCREMENTAL:
    # Process evolves gradually through several intermediate versions between start and end.
    log = generate_log_with_incremental_drift(
        num_versions      = 4,           # number of intermediate process versions
        traces            = None,        # split evenly if None (300 each)
        change_proportion = 0.1,         # evolution applied per step
        model             = make_tree(),
    )
    export_log(log, "logs/incremental_drift.xes")
    print("=== incremental ===")
    print(log.attributes['drift info'])
    print()
