import plotly.express as px
import plotly.graph_objects as go

def plot_activity_counts(stats: dict) -> go.Figure:
    counts = stats["activity_counts"]

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        labels={"x": "Count", "y": "Activity"},
        title="Activity Counts",
    )

    fig.update_yaxes(autorange="reversed")

    return fig

def plot_pca_variances(pca: PCA, cut_component: int = None) -> go.Figure:
    variances = stats["explained_var_per_component"]
    num_features = len(stats["loadings"].index)

    fig = px.bar(
        x=list(range(1, len(variances) + 1)),
        y=variances.values,
        labels={"x": "Component", "y": "Explained variance"},
        title=(
            f"PCA reduces {num_features}D \u2192 {stats['num_components']}D "
            f"({stats['explained_var']:.1%} variance explained)"
        ),
    )

    return fig

def plot_som_u_matrix():
    pass

def plot_som_heatmap():
    pass

def plot_som_colors():
    pass