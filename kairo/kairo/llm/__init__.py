import base64
import colorsys
import math
import struct
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from anthropic import Anthropic
from sklearn.decomposition import PCA

from ..analysis import DIVERGENCES, META, REFERENCES

class LLMConnector:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "EMPTY",
        base_url: str | None = None,
    ):
        self.provider = provider
    
        if provider == "anthropic":
            self.connector = _AnthropicConnector(
                model=model,
                api_key=api_key,
            )

        elif provider == "google":
            self.connector = _OpenAIConnector(
                model=model,
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

        elif provider == "openai":
            self.connector = _OpenAIConnector(
                model=model,
                api_key=api_key,
            )

        elif provider == "custom":
            self.connector = _OpenAIConnector(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def call(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        plots: list[go.Figure] | None = None,
        max_tokens: int = 1000,
        messages: list | None = None,
    ):
        # messages from count_input_tokens are sent as they are, without them they are
        # built here the same way from the prompt, the system prompt and the plots
        if messages is None:
            _, messages = count_input_tokens(self.provider, system_prompt, [prompt], plots or [])

        return self.connector.call(messages, max_tokens)


class _OpenAIConnector:
    def __init__(
        self,
        model: str,
        api_key: str = "EMPTY",
        base_url: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def call(self, messages: list, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )

class _AnthropicConnector:
    def __init__(
        self,
        model: str,
        api_key: str,
    ):
        self.model = model
        self.client = Anthropic(
            api_key=api_key,
        )

    def call(self, messages: list, max_tokens: int):
        # anthropic takes the system prompt apart from the messages
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [message for message in messages if message["role"] != "system"],
        }

        system = [message["content"] for message in messages if message["role"] == "system"]

        if system:
            kwargs["system"] = system[0]

        return self.client.messages.create(**kwargs)


# the system prompt the copilot starts with. it describes the approach and every step
# with its parameters, so keep it in line with the functions when they change

DEFAULT_SYSTEM_PROMPT = """You are the copilot of kairo, a Python library and Streamlit dashboard for state-based process monitoring, written for a bachelor thesis on concept drift detection in traditional event logs. You help the user read the results of the pipeline and decide what to inspect next.

THE APPROACH
An event log has one row per event, with at least a case id, an activity and a timestamp, and optionally a resource, an event id, a start timestamp and a duration. The behaviour of the process is described through states, and how often every state occurs is followed over time. When the process changes (concept drift), the mix of states it produces changes, and comparing that mix between calendar windows turns the change into a signal.

Three perspectives are planned: intra-case states (the situation of a single running case), resource states and inter-case states (the situation across all cases). Only the intra-case pipeline is built so far. It has five steps.

1. Features. Every event gets one row describing its case up to and including that event, its prefix. The feature groups are:
- act_freqs: the share of every activity among the case's events so far
- df_counts: how often every directly-follows pair of activities happened in the case so far
- act_set: 1 for every activity the case has already executed
- case_progress: the position of the event in its case, as a fraction of the case length
- current_act: 1 for the activity of the event itself
- past_acts: 1 for the activities of the previous events, one block per step back, as many steps as the sliding window size
The case id and the timestamp are carried along every step but never enter the maths.

2. Scaling and PCA. The features are scaled by z-score or by min-max to 0 to 1, the binary groups act_set, current_act and past_acts are left as they are. Z-scoring turns rare columns into large outliers, min-max avoids that. PCA is first fitted with a number of components, their explained variances show where to cut, and the components up to the cut are kept.

3. States. A clustering of the compressed rows gives every event a state.
- SOM: a grid of neurons, every cell (i, j) is a state and neighbouring cells hold similar states. Parameters: grid rows and columns, learning rate, distance (euclidean, cosine, manhattan, chebyshev). The u-matrix shows the distance of every neuron to its neighbours, dark ridges are borders between groups of states. Training is random, so cell numbers change between runs.
- k-means: k clusters under euclidean distance, the states are 0 to k-1.
- DBSCAN: dense regions become clusters. Parameters: eps (the neighbourhood radius), min_samples (rows within eps, itself included, that make a core row) and distance. Rows outside every cluster are noise, cluster -1, named "noise". The k-distance curve (the distance of every row to its min_samples-th neighbour, sorted) suggests eps at its knee. Prefix features repeat a lot, so the curve stays at 0 for most rows with the knee at its far right end, and DBSCAN tends to find many states.

4. Trajectories. A case moves through states over time. Consecutive events of a case in the same state form one visit, which lasts until the case's next visit starts. Trajectories are shown in calendar time, for a single case or for all cases whose first and last event lie in a date range, at most the 1,000 that started first.

5. Drift signal. The share of every state is counted per calendar window, a pandas frequency such as 12h, 1D, 7D or 30D (multi-day windows count from 1 January 1970, so weekly windows start on Thursdays). Every window's distribution is compared with a reference, the previous window, the mean of the lookback windows before it or the mean of all windows, by a divergence:
- KL divergence: unbounded, very sensitive to a state missing on one side
- Jensen-Shannon: bounded by ln 2, about 0.693
- total variation: 0 to 1, half the summed differences of the shares
- Hellinger: 0 to 1
A spike means the mix of states changed. One isolated spike points at a sudden drift, a stretch of raised scores at a gradual drift, repeated spikes at a recurring drift. Spikes at the very start or end of the log, around holidays or in windows with few events are often artefacts rather than drift.

KAIRO FUNCTIONS
Parameters with their defaults, dates are text like "2020-07-01" or timestamps, * is one of som, kmeans, dbscan.
- read_log(path, case_id, activity, timestamp, event_id=None, start_timestamp=None, resource=None, event_duration=None), compute_log_stats(log)
- compute_features_intra(log, features, sliding_window_size=0)
- standardize(feature_matrix, method "zscore" or "minmax", exclude=["current_act", "past_acts", "act_set"])
- compute_pca(feature_matrix, num_components=None), apply_pca(feature_matrix, pca, cut_component)
- compute_som(df, size=(5, 5), learning_rate=0.5, distance="euclidean"), get_som_winners(df, som)
- compute_kmeans(df, k=5), get_kmeans_clusters(df, kmeans)
- compute_dbscan(df, eps=0.5, min_samples=5, distance="euclidean"), get_dbscan_clusters(df, dbscan)
- get_som_state_distances(som, distance="euclidean"), get_kmeans_state_distances(kmeans), get_dbscan_state_distances(dbscan)
- get_*_state_frequencies(df, start_date=None, end_date=None), get_*_case_trajectory(df, case_id), get_*_trajectories(df, start_date=None, end_date=None, max_cases=1000)
- compute_state_distributions(df, window="7D"), compute_divergences(distributions, divergence="kl" | "js" | "tv" | "hellinger", reference="previous" | "recent" | "baseline", lookback=5)
- plots: plot_pca_variances, plot_som_u_matrix, plot_som_heatmap, plot_som_colors, plot_kmeans_frequencies, plot_dbscan_frequencies, plot_kmeans_distances, plot_dbscan_distances, plot_dbscan_k_distance, plot_*_case_trajectory, plot_*_trajectories, plot_state_distributions, plot_divergences
The dashboard runs these steps on its pages Features, PCA, States & Trajectories and Drift Signal, with the same parameters.

HOW TO ANSWER
- What you know about the log is only what the user shares: text summaries of the steps they ran and attached plots. Do not invent numbers, states, dates or cases.
- Name states as the summaries do: (i, j) for SOM cells, numbers for clusters, "noise" for DBSCAN noise. Name windows by their start dates and cases by their ids.
- When asked what to inspect, be concrete: date ranges around the strongest signals, cases whose trajectories pass through the states that changed, and parameters with values to try, such as another window, another divergence or reference, or more or fewer states.
- When the shared information is not enough, say so and name the step or plot that would answer it."""

def get_response_text(response) -> str:
    # the openai style apis answer with choices, anthropic with content blocks
    if hasattr(response, "choices"):
        return response.choices[0].message.content

    return "".join(block.text for block in response.content if block.type == "text")


def count_input_tokens(provider: str, system_prompt: str | None, prompts: list[str], plots: list[go.Figure]) -> tuple[int, list]:
    # the messages the provider's api takes, ready for LLMConnector.call, and about how many
    # tokens they are. text counts as 4 characters a token, images by the rule the
    # provider documents for their size in pixels
    text = "\n\n".join(prompts)
    images = [fig.to_image(format="png") for fig in plots]

    content = [{"type": "text", "text": text}]

    for image in images:
        data = base64.b64encode(image).decode("utf-8")

        if provider == "anthropic":
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})

    messages = [{"role": "user", "content": content}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    tokens = len((system_prompt or "") + text) // 4 + sum(_image_tokens(provider, image) for image in images)

    return tokens, messages

def count_output_tokens(provider: str, response) -> int:
    # the api reports it exactly, reasoning included
    if provider == "anthropic":
        return response.usage.output_tokens

    return response.usage.completion_tokens

def _image_tokens(provider: str, image: bytes) -> int:
    # the width and height of a png sit in its header
    width, height = struct.unpack(">II", image[16:24])

    if provider == "anthropic":
        # width * height / 750, bigger images are scaled down to about 1,600 tokens
        return min(width * height // 750, 1600)

    if provider == "google":
        # 258 tokens for every tile of 768 x 768
        return 258 * math.ceil(width / 768) * math.ceil(height / 768)

    # openai fits the image into 2048 x 2048, scales its short side down to 768, then counts
    # 170 tokens for every tile of 512 x 512 and 85 on top. local models all differ, so
    # this rule stands in for them too
    scale = min(1, 2048 / max(width, height))
    width, height = width * scale, height * scale
    scale = min(1, 768 / min(width, height))
    width, height = width * scale, height * scale

    return 85 + 170 * math.ceil(width / 512) * math.ceil(height / 512)


# abstractions: every step of the pipeline turned into text an llm can read

FEATURE_GROUPS = {
    "act_freqs": "share of every activity among the events of the case so far",
    "df_counts": "how often every directly-follows pair happened in the case so far",
    "act_set": "1 for every activity the case has already executed",
    "case_progress": "position of the event in its case, as a fraction of the case length",
    "current_act": "1 for the activity of the event itself",
    "past_acts": "1 for the activity of an earlier event, one block per step back",
}

def abstract_log_stats(stats: dict) -> str:
    lines = [
        f"Event log: {stats['events']:,} events of {stats['cases']:,} cases, {stats['activities']} "
        f"activities and {stats['resources']} resources, from {stats['start']} to {stats['end']}.",
        f"Throughput time per case in days: min {stats['tpt_days_min']:.2f}, mean {stats['tpt_days_mean']:.2f}, "
        f"median {stats['tpt_days_median']:.2f}, max {stats['tpt_days_max']:.2f}.",
        f"Events per case: min {stats['length_min']}, mean {stats['length_mean']:.1f}, "
        f"median {stats['length_median']:.1f}, max {stats['length_max']}.",
        "Events per activity: " + ", ".join(f"{a} ({n:,})" for a, n in stats["activity_counts"].items()),
    ]

    if len(stats["resource_counts"]):
        lines.append("Most active resources: " + ", ".join(
            f"{r} ({n:,})" for r, n in stats["resource_counts"].head(10).items()))

    return "\n".join(lines)

def abstract_features(features: pd.DataFrame) -> str:
    values = features.drop(columns=META, errors="ignore")
    groups = values.columns.str.split(":").str[0]

    lines = [
        f"Intra-case feature matrix: {len(features):,} rows, one per event, each describing "
        f"its case up to and including that event, with {values.shape[1]} feature columns.",
    ]

    if "case:concept:name" in features.columns:
        lines.append(f"The events belong to {features['case:concept:name'].nunique():,} cases.")

    if "time:timestamp" in features.columns:
        lines.append(f"They happened from {features['time:timestamp'].min()} to {features['time:timestamp'].max()}.")

    lines.append("Feature groups:")

    for group in dict.fromkeys(groups):
        block = values.loc[:, groups == group].to_numpy()
        means = values.loc[:, groups == group].mean().sort_values(ascending=False)
        meaning = next((text for name, text in FEATURE_GROUPS.items() if group.startswith(name)), "")

        lines.append(
            f"- {group} ({meaning}): {block.shape[1]} columns, mean value {block.mean():.3f}, "
            f"non-zero in {(block != 0).mean():.1%} of the cells, highest column means "
            + ", ".join(f"{column} = {mean:.3f}" for column, mean in means.head(3).items())
        )

    return "\n".join(lines)

def abstract_pca(pca: PCA, cut_component: int) -> str:
    ratios = pca.explained_variance_ratio_
    names = pca.feature_names_in_

    lines = [
        f"PCA fitted {pca.n_components_} components on {pca.n_features_in_} feature columns. "
        f"The cut keeps the first {cut_component}, which explain {ratios[:cut_component].sum():.1%} "
        f"of the variance.",
        "Explained variance per component: "
        + ", ".join(f"pc_{i + 1} {ratio:.1%}" for i, ratio in enumerate(ratios)),
        "Features with the strongest loadings on every kept component:",
    ]

    for i in range(cut_component):
        loadings = pca.components_[i]
        top = np.argsort(-np.abs(loadings))[:4]

        lines.append(f"- pc_{i + 1}: " + ", ".join(f"{names[k]} ({loadings[k]:+.2f})" for k in top))

    return "\n".join(lines)

def abstract_states(frequencies: pd.Series, distances: pd.DataFrame, color_mapping: dict) -> str:
    # frequencies is what get_som_state_frequencies, get_kmeans_state_frequencies or
    # get_dbscan_state_frequencies return
    total = frequencies.sum()
    never = [state for state in distances.index if state not in frequencies.index]

    lines = [f"{len(frequencies)} states occur among the {total:,} events. Events per state:"]
    lines += [
        f"- {state}: {count:,} events ({count / total:.1%})"
        for state, count in frequencies.sort_values(ascending=False).items()
    ]

    if never:
        lines.append("States no event landed in: " + ", ".join(never))

    lines.append("Nearest and farthest other state for every state, by the distance between them:")

    for state in distances.index:
        others = distances.loc[state].drop(state).sort_values()
        lines.append(
            f"- {state}: nearest {others.index[0]} ({others.iloc[0]:.2f}), "
            f"farthest {others.index[-1]} ({others.iloc[-1]:.2f})"
        )

    lines.append("Colors of the states in the plots:")

    for key, color in color_mapping.items():
        state = f"({key[0]}, {key[1]})" if isinstance(key, tuple) else ("noise" if key == -1 else str(key))
        lines.append(f"- {state}: {color} ({_color_name(color)})")

    return "\n".join(lines)

def abstract_case_trajectory(visits: pd.DataFrame, case_id: str) -> str:
    lines = [f"Trajectory of case {case_id}, one line per visit of a state, in order:"]
    lines += [
        f"- {visit.state} from {visit.start} to {visit.end} ({visit.duration}, {visit.events} events)"
        for visit in visits.itertuples()
    ]

    return "\n".join(lines)

def abstract_case_trajectories(trajectories: pd.DataFrame) -> str:
    # trajectories is what get_som_trajectories, get_kmeans_trajectories or
    # get_dbscan_trajectories return. hundreds of cases do not fit into a prompt one by one,
    # so their paths through the states are summarised
    cases = trajectories.groupby("case")
    paths = cases["state"].agg(" -> ".join)
    durations = cases["end"].max() - cases["start"].min()
    moves = (trajectories["state"] + " -> " + cases["state"].shift(-1)).dropna().value_counts()

    lines = [
        f"Trajectories of {len(paths):,} cases, from {trajectories['start'].min()} to {trajectories['end'].max()}. "
        f"A case visits {cases.size().mean():.1f} states on average and lasts {durations.median().round('s')} (median).",
        "The most common paths through the states, with how many cases took them:",
    ]
    lines += [f"- {path}: {count:,} cases ({count / len(paths):.1%})" for path, count in paths.value_counts().head(10).items()]

    lines.append("The most common moves from one state to the next:")
    lines += [f"- {move}: {count:,} times" for move, count in moves.head(10).items()]

    lines.append("States the cases start in: " + ", ".join(
        f"{state} ({count:,})" for state, count in cases["state"].first().value_counts().head(5).items()))
    lines.append("States the cases end in: " + ", ".join(
        f"{state} ({count:,})" for state, count in cases["state"].last().value_counts().head(5).items()))

    return "\n".join(lines)

def abstract_distributions(distributions: pd.DataFrame) -> str:
    # distributions is what compute_state_distributions returns
    changes = distributions.diff()
    # half the summed changes of the shares, how much of the mix moved since the window before
    moved = changes.abs().sum(axis=1) / 2

    lines = [
        f"State distribution over {len(distributions)} calendar windows from {distributions.index[0]} "
        f"to {distributions.index[-1]}: the share of every state among the events of each window.",
        "Share of every state over the windows, mean, lowest and highest:",
    ]

    for state in distributions.columns:
        shares = distributions[state]
        lines.append(
            f"- {state}: mean {shares.mean():.1%}, lowest {shares.min():.1%} in the window starting "
            f"{shares.idxmin()}, highest {shares.max():.1%} in the window starting {shares.idxmax()}"
        )

    lines.append("The largest changes from one window to the next, with the states that changed most:")

    for window, share in moved.nlargest(5).items():
        change = changes.loc[window]
        lines.append(
            f"- window starting {window}: {share:.1%} of the mix moved, "
            + ", ".join(f"{state} {change[state]:+.1%}" for state in change.abs().nlargest(3).index)
        )

    return "\n".join(lines)

def abstract_divergences(divergences: pd.DataFrame, divergence: str, reference: str, lookback: int = 5) -> str:
    # divergences is what compute_divergences returns for this divergence, reference and lookback
    against = f"the mean of the {lookback} windows before it" if reference == "recent" else f"the {REFERENCES[reference]}"
    scores = divergences["score"].dropna()

    lines = [
        f"Drift signal: the {DIVERGENCES[divergence]} between the state distribution of every calendar window "
        f"and {against}, over {len(divergences)} windows from {divergences['window'].iloc[0]} to "
        f"{divergences['window'].iloc[-1]}. The windows are numbered from 0 as in the plot.",
        f"Scores: median {scores.median():.3f}, mean {scores.mean():.3f}, highest {scores.max():.3f}.",
        "The windows with the highest scores:",
    ]
    lines += [
        f"- window {number}, starting {row['window']}: {row['score']:.3f}"
        for number, row in divergences.dropna().nlargest(5, "score").iterrows()
    ]

    return "\n".join(lines)

def _color_name(color: str) -> str:
    # a rough name, so the llm can match "the purple bar" in a plot to its state
    r, g, b = (int(color[k:k + 2], 16) / 255 for k in (1, 3, 5))
    hue, saturation, _ = colorsys.rgb_to_hsv(r, g, b)

    if saturation < 0.2:
        return "grey"

    hues = [(0.04, "red"), (0.11, "orange"), (0.19, "yellow"), (0.45, "green"), (0.54, "cyan"),
            (0.72, "blue"), (0.83, "purple"), (0.95, "pink"), (1.0, "red")]

    return next(name for bound, name in hues if hue <= bound)
