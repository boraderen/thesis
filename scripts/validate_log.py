from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

from scripts.driftify.config import (
    ACTIVITY_KEY,
    AMOUNT_KEY,
    CASE_ID_KEY,
    CASE_TYPE_KEY,
    DURATION_KEY,
    EVENT_ID_KEY,
    GeneratorConfig,
    LEGACY_TO_NATIVE_COLUMNS,
    REGION_KEY,
    RESOURCE_KEY,
    ROLE_KEY,
    SCHEMA_COLUMNS,
    START_TIMESTAMP_KEY,
    TIMESTAMP_KEY,
)


Issue = dict[str, Any]


def validate_xes(path: str | Path, config: GeneratorConfig | None = None) -> list[Issue]:
    xes_path = Path(path)
    issues: list[Issue] = []
    sidecar = _load_sidecar(xes_path)
    config_data = sidecar.get("config", {}) if sidecar else {}
    cfg = config or _config_from_metadata(config_data)

    log = _read_xes(xes_path, issues)
    if log is None:
        return issues
    df = _event_log_to_dataframe(log)
    df = _canonicalize_columns(df)
    _check_schema(df, issues)
    if any(issue["severity"] == "error" for issue in issues):
        return issues
    _check_types(df, issues)
    _check_timestamps(df, issues)
    _check_drift_metadata(df, log, sidecar, issues)
    _check_variant_metadata(df, sidecar, issues)
    if cfg is not None:
        _check_config_counts(df, cfg, issues)
        _check_distribution_stability(df, sidecar, issues)
    return issues


def validation_passed(issues: list[Issue], *, strict: bool = False) -> bool:
    severities = {"error"} if not strict else {"error", "warning"}
    return not any(issue["severity"] in severities for issue in issues)


def _read_xes(path: Path, issues: list[Issue]):
    if not path.exists():
        issues.append(_issue("error", "file_missing", f"{path} does not exist"))
        return None
    try:
        ET.parse(path)
    except ET.ParseError as exc:
        issues.append(_issue("error", "xes_not_well_formed", str(exc)))
        return None
    try:
        log = xes_importer.apply(str(path))
    except Exception as exc:  # pragma: no cover - exact PM4PY exceptions vary
        issues.append(_issue("error", "pm4py_roundtrip_failed", str(exc)))
        return None
    if len(log) == 0:
        issues.append(_issue("error", "empty_log", "PM4PY imported zero traces"))
    return log


def _event_log_to_dataframe(log) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    native_to_legacy = {native: legacy for legacy, native in LEGACY_TO_NATIVE_COLUMNS.items()}
    for trace in log:
        fallback_case_id = trace.attributes.get(CASE_ID_KEY, trace.attributes.get("concept:name", ""))
        for event in trace:
            row = {
                column: event.get(column, event.get(native_to_legacy.get(column, "")))
                for column in SCHEMA_COLUMNS
            }
            if row.get(CASE_ID_KEY) in {None, ""}:
                row[CASE_ID_KEY] = fallback_case_id
            for column in [CASE_TYPE_KEY, AMOUNT_KEY, REGION_KEY]:
                if row.get(column) in {None, ""} and column in trace.attributes:
                    row[column] = trace.attributes[column]
                legacy_column = native_to_legacy.get(column, "")
                if row.get(column) in {None, ""} and legacy_column in trace.attributes:
                    row[column] = trace.attributes[legacy_column]
            rows.append(row)
    return pd.DataFrame(rows)


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={old: new for old, new in LEGACY_TO_NATIVE_COLUMNS.items() if old in df.columns})


def _check_schema(df: pd.DataFrame, issues: list[Issue]) -> None:
    missing = [column for column in SCHEMA_COLUMNS if column not in df.columns]
    if missing:
        issues.append(_issue("error", "schema_missing_columns", "Required columns are absent", columns=missing))
    extra = [column for column in df.columns if column not in SCHEMA_COLUMNS]
    if extra:
        issues.append(_issue("warning", "schema_extra_columns", "Extra columns were found", columns=extra))


def _check_types(df: pd.DataFrame, issues: list[Issue]) -> None:
    string_columns = [EVENT_ID_KEY, CASE_ID_KEY, ACTIVITY_KEY, RESOURCE_KEY, ROLE_KEY, CASE_TYPE_KEY, REGION_KEY]
    for column in string_columns:
        if df[column].isna().any() or not df[column].map(lambda value: isinstance(value, str)).all():
            issues.append(_issue("error", "column_type", f"{column} must contain strings", column=column))
    for column in [START_TIMESTAMP_KEY, TIMESTAMP_KEY]:
        converted = pd.to_datetime(df[column], utc=True, errors="coerce")
        if converted.isna().any():
            issues.append(_issue("error", "column_type", f"{column} must contain timestamps", column=column))
    for column in [DURATION_KEY, AMOUNT_KEY]:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().any():
            issues.append(_issue("error", "column_type", f"{column} must contain numeric values", column=column))


def _check_timestamps(df: pd.DataFrame, issues: list[Issue]) -> None:
    timestamps = pd.to_datetime(df[TIMESTAMP_KEY], utc=True, errors="coerce")
    starts = pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True, errors="coerce")
    if (starts > timestamps).any():
        issues.append(_issue("error", "start_after_completion", f"{START_TIMESTAMP_KEY} is after {TIMESTAMP_KEY}"))
    expected = (timestamps - starts).dt.total_seconds() / 60.0
    actual = pd.to_numeric(df[DURATION_KEY], errors="coerce")
    mismatch = (expected - actual).abs() > 1e-6
    if mismatch.any():
        issues.append(
            _issue(
                "error",
                "duration_mismatch",
                f"{DURATION_KEY} does not match timestamp difference",
                rows=int(mismatch.sum()),
            )
        )
    for case_id, group in df.assign(_timestamp=timestamps).groupby(CASE_ID_KEY, sort=False):
        if not group["_timestamp"].is_monotonic_increasing:
            issues.append(
                _issue(
                    "error",
                    "case_timestamps_not_monotonic",
                    "timestamps must be non-decreasing within each case",
                    case_id=str(case_id),
                )
            )


def _check_drift_metadata(df: pd.DataFrame, log, sidecar: dict[str, Any], issues: list[Issue]) -> None:
    drift_info = log.attributes.get("drift_info")
    if drift_info is None:
        issues.append(_issue("error", "drift_info_missing", "log-level drift_info attribute is absent"))
        drifts = sidecar.get("drifts", []) if sidecar else []
    else:
        try:
            drifts = json.loads(drift_info)
        except json.JSONDecodeError as exc:
            issues.append(_issue("error", "drift_info_invalid_json", str(exc)))
            drifts = sidecar.get("drifts", []) if sidecar else []
    if sidecar and "drifts" in sidecar and drift_info is not None:
        if json.dumps(sidecar["drifts"], sort_keys=True, separators=(",", ":")) != json.dumps(drifts, sort_keys=True, separators=(",", ":")):
            issues.append(_issue("warning", "metadata_sidecar_mismatch", "sidecar and log-level drift metadata differ"))
    if not drifts:
        return
    start = pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True).min()
    end = pd.to_datetime(df[TIMESTAMP_KEY], utc=True).max()
    required = {
        "drift_id",
        "perspective",
        "subtype",
        "drift_type",
        "change_point_timestamp",
        "overlap_window_start",
        "overlap_window_end",
        "affected_columns",
        "change_details",
    }
    for drift in drifts:
        missing = sorted(required.difference(drift))
        if missing:
            issues.append(_issue("error", "drift_metadata_missing_fields", "drift metadata is incomplete", missing=missing))
            continue
        change_point = pd.to_datetime(drift["change_point_timestamp"], utc=True)
        if change_point < start or change_point > end:
            issues.append(
                _issue(
                    "error",
                    "drift_change_point_outside_log",
                    "change point is outside the log time range",
                    drift_id=drift["drift_id"],
                    change_point=str(change_point),
                    log_start=str(start),
                    log_end=str(end),
                )
            )


def _check_variant_metadata(df: pd.DataFrame, sidecar: dict[str, Any], issues: list[Issue]) -> None:
    if not sidecar:
        return
    config = sidecar.get("config", {})
    if "num_trace_variants" in config:
        expected = _trace_variant_count(df)
        actual = int(config["num_trace_variants"])
        if actual != expected:
            issues.append(
                _issue(
                    "error",
                    "variant_count_mismatch",
                    "metadata num_trace_variants does not match observed trace variants",
                    metadata_value=actual,
                    observed_value=expected,
                )
            )
    deprecated_keys = {
        "observed_variant_count_before",
        "observed_variant_count_after",
        "observed_variant_counts_by_version",
        "structural_variant_estimate_before",
        "structural_variant_estimate_after",
        "structural_variant_estimates_by_version",
        "variant_counts",
    }
    for drift in sidecar.get("drifts", []):
        details = drift.get("change_details", {})
        found = sorted(deprecated_keys.intersection(details))
        if found:
            issues.append(
                _issue(
                    "warning",
                    "deprecated_variant_metadata",
                    "control-flow variant metadata should contain only observed variant_count fields",
                    drift_id=drift.get("drift_id"),
                    fields=found,
                )
            )


def _check_config_counts(df: pd.DataFrame, config: GeneratorConfig, issues: list[Issue]) -> None:
    allowed = {
        RESOURCE_KEY: set(config.resources),
        ROLE_KEY: set(config.roles),
        CASE_TYPE_KEY: set(config.case_types),
        REGION_KEY: set(config.regions),
        ACTIVITY_KEY: set(config.activity_pool),
    }
    for column, values in allowed.items():
        observed = set(df[column].dropna().astype(str))
        unknown = sorted(observed.difference(values))
        if unknown:
            issues.append(_issue("error", "unexpected_values", f"{column} contains values outside config", column=column, values=unknown))
    max_counts = {
        RESOURCE_KEY: config.num_resources,
        ROLE_KEY: config.num_roles,
        CASE_TYPE_KEY: config.num_case_types,
        REGION_KEY: len(config.regions),
        ACTIVITY_KEY: config.max_activities,
    }
    for column, maximum in max_counts.items():
        observed_count = df[column].nunique()
        if observed_count > maximum:
            issues.append(
                _issue(
                    "error",
                    "count_exceeds_config",
                    f"{column} has more distinct values than allowed",
                    column=column,
                    observed=int(observed_count),
                    expected_max=int(maximum),
                )
            )


def _check_distribution_stability(df: pd.DataFrame, sidecar: dict[str, Any], issues: list[Issue]) -> None:
    drifts = sidecar.get("drifts", []) if sidecar else []
    if any(drift.get("perspective") == "resource" for drift in drifts):
        return
    if df[CASE_ID_KEY].nunique() < 40:
        issues.append(_issue("info", "stability_check_skipped", "too few traces for distribution stability check"))
        return
    ordered_cases = (
        df[[CASE_ID_KEY, START_TIMESTAMP_KEY]]
        .assign(**{START_TIMESTAMP_KEY: pd.to_datetime(df[START_TIMESTAMP_KEY], utc=True)})
        .groupby(CASE_ID_KEY, as_index=False)[START_TIMESTAMP_KEY]
        .min()
        .sort_values(START_TIMESTAMP_KEY)
    )
    third = len(ordered_cases) // 3
    if third < 10:
        issues.append(_issue("info", "stability_check_skipped", "too few traces per temporal segment"))
        return
    early = set(ordered_cases.head(third)[CASE_ID_KEY])
    late = set(ordered_cases.tail(third)[CASE_ID_KEY])
    early_df = df[df[CASE_ID_KEY].isin(early)]
    late_df = df[df[CASE_ID_KEY].isin(late)]
    distance = _mean_activity_resource_tvd(early_df, late_df)
    if distance > 0.65:
        issues.append(_issue("error", "unstable_resource_distribution", "activity-resource distribution changed without resource drift", distance=round(distance, 4)))
    elif distance > 0.45:
        issues.append(_issue("warning", "unstable_resource_distribution", "activity-resource distribution may have changed without resource drift", distance=round(distance, 4)))


def _mean_activity_resource_tvd(early: pd.DataFrame, late: pd.DataFrame) -> float:
    distances = []
    resources = sorted(set(early[RESOURCE_KEY]).union(late[RESOURCE_KEY]))
    for activity in sorted(set(early[ACTIVITY_KEY]).intersection(late[ACTIVITY_KEY])):
        left = _normalized_counts(early.loc[early[ACTIVITY_KEY] == activity, RESOURCE_KEY], resources)
        right = _normalized_counts(late.loc[late[ACTIVITY_KEY] == activity, RESOURCE_KEY], resources)
        distances.append(float(0.5 * np.abs(left - right).sum()))
    return float(np.mean(distances)) if distances else 0.0


def _normalized_counts(values: pd.Series, universe: list[str]) -> np.ndarray:
    counts = Counter(values.astype(str))
    total = max(1, sum(counts.values()))
    return np.array([counts[value] / total for value in universe], dtype=float)


def _trace_variant_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    variants = {
        tuple(group.sort_values([START_TIMESTAMP_KEY, TIMESTAMP_KEY])[ACTIVITY_KEY].astype(str))
        for _, group in df.groupby(CASE_ID_KEY, sort=False)
    }
    return len(variants)


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".metadata.json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _config_from_metadata(config_data: dict[str, Any]) -> GeneratorConfig | None:
    if not config_data:
        return None
    field_names = set(GeneratorConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in config_data.items() if key in field_names and key != "start_timestamp"}
    return GeneratorConfig(**kwargs)


def _issue(severity: str, code: str, message: str, **details: Any) -> Issue:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a driftify XES log.")
    parser.add_argument("paths", nargs="+", help="XES file(s) to validate")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args(argv)

    all_results: dict[str, list[Issue]] = {}
    exit_code = 0
    for path in args.paths:
        issues = validate_xes(path)
        all_results[path] = issues
        if not validation_passed(issues, strict=args.strict):
            exit_code = 1
    if args.json:
        print(json.dumps(all_results, indent=2, sort_keys=True))
    else:
        for path, issues in all_results.items():
            status = "PASS" if validation_passed(issues, strict=args.strict) else "FAIL"
            print(f"{path}: {status}")
            for issue in issues:
                print(f"  [{issue['severity']}] {issue['code']}: {issue['message']} {issue['details']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
