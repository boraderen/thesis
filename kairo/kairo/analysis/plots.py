import plotly.express as px

def plot_activity_counts(stats: dict):
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