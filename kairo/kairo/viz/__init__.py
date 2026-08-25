"""Figures. Every function returns a plotly Figure and renders nothing."""
from .palette import DISTINCT_COLORS, state_bg, state_color, state_fg
from .plots import (
    add_transition_markers,
    add_window_boundaries,
    plot_activity_frequency,
    plot_drift_signal,
    plot_k_distance,
    plot_pca_variance,
    plot_state_distances,
    plot_state_distribution,
    plot_state_grid,
    plot_trajectory,
    plot_transition_matrix,
    save_figure,
)
