from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DriftMetadata:
    drift_id: str
    perspective: str
    subtype: str
    drift_type: str
    change_point_timestamp: datetime
    overlap_window_start: datetime | None
    overlap_window_end: datetime | None
    affected_columns: list[str]
    change_details: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "perspective": self.perspective,
            "subtype": self.subtype,
            "drift_type": self.drift_type,
            "change_point_timestamp": self.change_point_timestamp.isoformat(),
            "overlap_window_start": self.overlap_window_start.isoformat()
            if self.overlap_window_start
            else None,
            "overlap_window_end": self.overlap_window_end.isoformat()
            if self.overlap_window_end
            else None,
            "affected_columns": self.affected_columns,
            "change_details": self.change_details,
        }


def metadata_payload(
    *,
    log_name: str,
    config: dict[str, Any],
    drifts: list[DriftMetadata],
    noise: dict[str, Any],
) -> dict[str, Any]:
    return {
        "log_name": log_name,
        "config": config,
        "drifts": [drift.to_json_dict() for drift in drifts],
        "noise": noise,
    }


def drift_info_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload["drifts"], sort_keys=True, separators=(",", ":"))
