"""Feature extraction: one FeatureSet shape for all three perspectives."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    """A computed feature matrix plus everything needed to interpret it.

    `matrix` holds only feature columns; `index` holds the identifying columns
    (case / activity / timestamp for intra-case, window_start for the windowed
    perspectives). `groups` maps each computed feature key to its columns.
    """

    perspective: str
    matrix: pd.DataFrame
    index: pd.DataFrame
    groups: dict[str, list[str]]
    meta: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.matrix.columns.tolist()

    @property
    def frame(self) -> pd.DataFrame:
        """Index and feature columns side by side, as the dashboard shows them."""
        return pd.concat([self.index.reset_index(drop=True), self.matrix.reset_index(drop=True)], axis=1)

    def values(self, columns: list[str] | None = None) -> np.ndarray:
        return (self.matrix if columns is None else self.matrix[columns]).to_numpy(dtype=float)


def select_columns(
    fs: FeatureSet,
    features: list[str] | None = None,
    resources: list[str] | None = None,
    activities: list[str] | None = None,
) -> list[str]:
    """Filter a FeatureSet's columns by feature keys and, where they apply, by
    the resources / activities a column refers to."""
    keys = list(fs.groups) if features is None else [k for k in features if k in fs.groups]
    selected: list[str] = []
    for key in keys:
        for col in fs.groups[key]:
            if resources is not None and ":" in col:
                if key == "ho":
                    pair = col.split(":", 1)[1]
                    src, dst = pair.split("→")
                    if src not in resources or dst not in resources:
                        continue
                elif key == "activity_events":
                    _, activity, resource = col.split(":", 2)
                    if resource not in resources:
                        continue
                    if activities is not None and activity not in activities:
                        continue
                elif key in ("events", "active", "duration", "wait"):
                    if col.split(":", 1)[1] not in resources:
                        continue
            selected.append(col)
    return selected


def build_features(log: pd.DataFrame, perspective: str, **kwargs) -> FeatureSet:
    """Build the feature matrix of a perspective; kwargs go to that builder.

    intra_case: features, history · resource: features, window_minutes,
    resources, activities · inter_case: features, window_minutes,
    stall_minutes, numeric_attrs, categorical_attrs.
    """
    from . import inter, intra, resource

    builders = {"intra_case": intra.build, "resource": resource.build, "inter_case": inter.build}
    if perspective not in builders:
        raise ValueError(f"Unknown perspective: {perspective!r} (use one of {tuple(builders)})")
    return builders[perspective](log, **kwargs)
