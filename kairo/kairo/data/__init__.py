"""The event log itself: canonical schema, ingestion, statistics, and calendar windows."""
from .log import (
    LogStatistics,
    classify_attributes,
    log_statistics,
    map_columns,
    read_log,
    span_label,
)
from .schema import (
    FEATURE_DESCRIPTIONS,
    FEATURE_LABELS,
    feature_availability,
    missing_columns,
    requirements_table,
)
from .windows import (
    as_window_minutes,
    default_window_minutes,
    log_span_minutes,
    window_minute_choices,
    window_minute_label,
)
