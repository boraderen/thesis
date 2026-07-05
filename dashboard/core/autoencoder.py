"""Pretrained intra-case autoencoder: prefix features → fixed 441-D space → latent.

The model and scaler are trained once in notebooks/train_autoencoder.ipynb on many
rheon logs over a fixed alphabet of 20 activity slots (a..t) and saved under
data/autoencoder/. Any log with at most 20 distinct activities can be encoded
without retraining: each activity is assigned one of the fixed slots (identity for
logs already named within a..t, sorted order otherwise), its feature columns are
mapped onto the training columns (zeros for unused slots), scaled with the training
scaler, and pushed through the encoder. The latent vectors replace PCA scores as
the SOM input; the per-event reconstruction error says how familiar each event
looks to the model.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from core.features.intra_case import IntraSpec

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "autoencoder"

AE_ACTIVITIES = list(string.ascii_lowercase[:20])  # the alphabet the model was trained on
AE_FEATURE_COLS = (
    [f"freq_{a}" for a in AE_ACTIVITIES]
    + [f"bigram_{a}->{b}" for a in AE_ACTIVITIES for b in AE_ACTIVITIES]
    + [f"seen_{a}" for a in AE_ACTIVITIES]
    + ["progress_ratio"]
)


class _Autoencoder(nn.Module):
    """Same architecture as in notebooks/train_autoencoder.ipynb."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass(frozen=True)
class AutoencoderResult:
    """Latent SOM input plus reconstruction diagnostics of the pretrained model."""

    transformed: np.ndarray    # (n_events, latent_dim) encoder output — the SOM input
    recon_errors: np.ndarray   # (n_events,) per-event MSE in the scaled feature space
    explained_variance: float  # reconstruction R² = 1 - SSE/SST on this log; comparable to PCA's cumulative ratio
    raw_dim: int
    latent_dim: int
    n_parameters: int


def artifacts_available() -> bool:
    """Whether the trained model + scaler exist under data/autoencoder/."""
    return (ARTIFACT_DIR / "autoencoder.pt").exists() and (ARTIFACT_DIR / "scaler.joblib").exists()


def _activity_slots(activities: list[str]) -> dict[str, str]:
    """Assign each activity one of the fixed a..t slots.

    Logs already named within a..t keep their names (the rheon convention the
    model was trained on); any other naming is slotted in sorted order.
    """
    if len(activities) > len(AE_ACTIVITIES):
        raise ValueError(
            f"log has {len(activities)} distinct activities; "
            f"the autoencoder supports at most {len(AE_ACTIVITIES)}"
        )
    if set(activities) <= set(AE_ACTIVITIES):
        return {activity: activity for activity in activities}
    return {activity: AE_ACTIVITIES[i] for i, activity in enumerate(sorted(activities))}


@st.cache_resource(show_spinner=False)
def _load_artifacts() -> tuple[_Autoencoder, object]:
    checkpoint = torch.load(ARTIFACT_DIR / "autoencoder.pt", map_location="cpu")
    model = _Autoencoder(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    return model, scaler


def fixed_matrix(feat: pd.DataFrame, spec: IntraSpec) -> np.ndarray:
    """Map the log-specific feature columns onto the fixed training columns."""
    slot = _activity_slots(spec.activities)
    rename = {"progress_ratio": "progress_ratio"}
    for activity in spec.activities:
        rename[f"activity_freq:{activity}"] = f"freq_{slot[activity]}"
        rename[f"vocab:{activity}"] = f"seen_{slot[activity]}"
    for src, dst in spec.transitions:
        rename[f"bigram:{src}→{dst}"] = f"bigram_{slot[src]}->{slot[dst]}"
    fixed = feat[list(rename)].rename(columns=rename)
    return fixed.reindex(columns=AE_FEATURE_COLS, fill_value=0.0).to_numpy(dtype=np.float32)


@st.cache_data(show_spinner=False)
def encode_matrix(matrix: np.ndarray) -> AutoencoderResult:
    """Scale with the training scaler, encode to latent vectors, score reconstruction."""
    model, scaler = _load_artifacts()
    scaled = torch.from_numpy(scaler.transform(matrix).astype(np.float32))
    with torch.no_grad():
        recon, latent = model(scaled)
    errors = ((recon - scaled) ** 2).mean(dim=1).numpy()
    sse = float(((recon - scaled) ** 2).sum())
    sst = float(((scaled - scaled.mean(dim=0)) ** 2).sum())
    return AutoencoderResult(
        transformed=latent.numpy(),
        recon_errors=errors,
        explained_variance=1.0 - sse / sst if sst > 0 else float("nan"),
        raw_dim=matrix.shape[1],
        latent_dim=latent.shape[1],
        n_parameters=sum(p.numel() for p in model.parameters()),
    )
