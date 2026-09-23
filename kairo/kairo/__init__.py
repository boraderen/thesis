from .data import (
    read_log,
    compute_log_stats,
)

from .analysis import (
    standardize,
    compute_pca,
    apply_pca,
    compute_som,
    get_som_winners,
)

from .analysis.intra import (
    compute_features_intra,
)

from .analysis.plots import (
    plot_activity_counts,
    plot_pca_variances,
    plot_som_u_matrix,
    plot_som_heatmap,
    compute_som_color_mapping,
    plot_som_colors,
)

from .llm import (
    LLMConnector,
)

__version__ = "0.1.0"
