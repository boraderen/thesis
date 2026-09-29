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

from ..analysis import DISTANCES, DIVERGENCES, META, REFERENCES
from ..analysis.resource import FEATURES as RESOURCE_FEATURES

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
        # messages from create_messages are sent as they are, without them they are
        # built here the same way from the prompt, the system prompt and the plots
        if messages is None:
            messages = create_messages(self.provider, system_prompt, [prompt], plots or [])

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
        # anthropic takes the system prompt apart from the messages. the answer is streamed,
        # the sdk refuses to wait for more than about 21,000 tokens otherwise
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [message for message in messages if message["role"] != "system"],
        }

        system = [message["content"] for message in messages if message["role"] == "system"]

        if system:
            kwargs["system"] = system[0]

        with self.client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()


# the system prompt the copilot starts with. it explains the method, the features and what
# kairo can compute in plain words, so keep it in line with the pipeline when it changes

DEFAULT_SYSTEM_PROMPT = """You are the analysis assistant of kairo, a framework for state-based process monitoring and concept drift detection in traditional event logs. In a chat, you help the user find out whether, when and how their process changed: you read every detail they share, answer and clarify their questions, and ask for what is missing.

THE METHOD
An event log has one row per event with a case id, an activity and a timestamp, and optionally a resource, durations and case attributes. Kairo describes the process through states and follows how often every state occurs over time. When the process changes, the mix of states changes, and comparing that mix between calendar windows gives a drift signal. Three perspectives run the same pipeline:
- Intra-case: the situation of a single running case, one row per event.
- Resource: how the resources work, one row per calendar window (e.g. 1D).
- Inter-case: the situation across all running cases, one row per calendar window.
The pipeline: features, then optional scaling (z-score, or min-max to 0-1) and optional PCA (fit components, cut where the explained variance flattens), then clustering into states (SOM grid cells, k-means clusters, or DBSCAN dense regions with noise), then trajectories through the states, then the drift signal: the share of every state per window (larger than the feature window for resource and inter-case) compared to a reference (previous window, mean of the recent windows, or mean of all windows) by KL (unbounded, explodes when a state is missing on one side), Jensen-Shannon (at most ln 2), total variation (0-1) or Hellinger (0-1). Resource and inter-case also have window distances: the distance between every window's feature vector and the same kind of reference, without clustering.

THE FEATURES
Intra-case, describing the case up to and including the event (its prefix):
- act_freqs: share of every activity among the case's events so far
- df_counts: how often every directly-follows pair of activities occurred so far
- act_set: which activities the case has already executed
- case_progress: position of the event as a fraction of its case's length
- current_act: the activity of the event itself
- past_acts: the activities of the last few events, as many as the sliding window size
Resource, per resource (or pair) in a window:
- res_events, res_cases: events executed and distinct cases touched
- res_durations: mean event duration in minutes
- res_waits: mean minutes a case waited for the resource after another resource's event (handover waits)
- act_res_shares: for every activity, the share of its events executed by each resource
- handover_shares: for every resource, the share of its handovers going to each other resource
Inter-case, over all cases in a window:
- active_cases, new_arrivals, completions: cases with an event, starting, and ending in the window
- events_per_case: events per active case
- mean_delta_t, std_delta_t: mean and spread of the minutes between consecutive events of a case
- stalled_cases: running cases whose last event is older than the stall threshold
- attr_means, attr_stds, attr_shares: numerical case attributes' mean and spread, categorical ones' value shares

WHAT KAIRO CAN PROVIDE
Log statistics; the features; PCA explained variances and the strongest features per component; states with their sizes, distances between them and colors; SOM u-matrix and heatmaps; DBSCAN k-distance curve for picking eps; state frequencies for any date range, e.g. before and after a suspected drift; the trajectory of a single case, of all cases in a date range, or of the log window by window; state distributions per window; divergences and window distances per window; and clustering scores: silhouette (-1 to 1, above about 0.5 is good, below 0.25 weak), Calinski-Harabasz (higher is better, only comparable on the same rows), DBCV for DBSCAN (-1 to 1, noise counts against it), SOM quantization error (lower is closer, but bigger grids always lower it) and SOM occupancy entropy (0 to 1, how evenly the grid is used). Everything can be rerun with other windows, features, scaling, PCA cuts, methods, parameters, divergences, references and date ranges.

STRENGTHS AND LIMITS
- Intra-case features are pure control flow: they see new, missing or reordered activities, loops and skips, but no time, resources or attributes. A process that only got slower is invisible there.
- Resource features see who does what, workload, durations, waits and handovers, but not the order of activities within a case.
- Inter-case features see load, arrivals, pace, stalls and case mix, but not which activities or resources.
- So drifts do not always spread. A control-flow change shows in intra-case and often, but not always, in resource shares or events per case. A staffing or reassignment change shows in resource and maybe in inter-case pace, not in intra-case. A workload or arrival change shows in inter-case and resource volume, not in intra-case. A slowdown shows in inter-case and resource times only. Expect a perspective to react only when its features can see the change, and do not read a quiet perspective as evidence against a drift it cannot see.
- States are unsupervised and depend on every earlier choice. SOM cell numbers change between runs, k-means favours round equal clusters, DBSCAN on intra-case prefixes (many identical rows) tends to find many states.
- Window perspectives have only as many rows as windows, often a few hundred, so their clusterings are fragile. Empty or quiet windows (weekends, holidays) can form their own state and look like a recurring drift.
- Signals at the very start or end of the log (truncated cases), in windows with few events, or from KL on a state missing in one window are often artefacts.
- The signal tells when the mix changed, not why. The why comes from which states changed and what they stand for.

HOW TO WORK
- Use only what the user shares: summaries and plots. Never invent numbers, states, dates, cases or resources, and name them as the summaries do ((i, j) for SOM cells, numbers for clusters, "noise"; windows by start date).
- Read every detail and connect them: which states grow or shrink around a spike, what their features mean, whether the timing fits the event counts.
- Judge the configuration before the signal. Say plainly when it is not meaningful and why, e.g. one state holding almost all rows, many empty SOM cells (low occupancy entropy), weak silhouette, DBSCAN mostly noise or one giant cluster, a PCA cut keeping little variance or dominated by a few rare z-scored columns, drift windows too small for the event volume or not larger than the feature windows, features that cannot see the suspected change. Then suggest a concrete better configuration, for one or all perspectives, with values.
- Be skeptical of drift signals. Trust a drift when it is consistent: the same dates across divergences, references and windows, a clear shift in states that stand for real behaviour, and agreement between perspectives that can see it. When results contradict each other or could be artefacts, say so and name the one or two checks that would settle it (a date filter around the spike, frequencies before and after, a case or log trajectory, another window, divergence, method or parameters). When the evidence is clear, state the drift directly: the date range, the type (sudden: one isolated spike; gradual: a stretch of raised scores; recurring: repeated spikes), the perspectives it shows in, and what changed.
- Ask for more only when it would change the conclusion, and do not repeat suggestions already made or ask for things already shared. Ask a clarifying question when the user's goal is unclear.
- Keep answers compact and in plain language. Explain what results mean, not how to call functions."""

def get_response_text(response) -> str:
    # the openai style apis answer with choices, anthropic with content blocks
    if hasattr(response, "choices"):
        return response.choices[0].message.content

    return "".join(block.text for block in response.content if block.type == "text")


def get_chat_history(chat_history: list) -> str:
    # the messages sent so far and the answers as readable text, one block per message.
    # the system prompt is cut to its size and every plot stands in as a placeholder
    blocks = []

    for message in chat_history:
        content = message["content"]

        if message["role"] == "system":
            text = f"({len(content):,} characters)"
        elif isinstance(content, str):
            text = content
        else:
            texts = [block["text"] for block in content if block["type"] == "text"]
            plots = len(content) - len(texts)
            text = "\n\n".join(texts) + (f"\n\n[{plots} plot{'s' if plots != 1 else ''}]" if plots else "")

        blocks.append(f"── {message['role']} ──\n{text.strip()}")

    return "\n\n".join(blocks)

def create_messages(provider: str, system_prompt: str | None, prompts: list[str], plots: list[go.Figure]) -> list:
    # the messages the provider's api takes, ready for LLMConnector.call and count_input_tokens
    text = "\n\n".join(prompts)
    content = [{"type": "text", "text": text}]

    for fig in plots:
        data = base64.b64encode(fig.to_image(format="png")).decode("utf-8")

        if provider == "anthropic":
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})

    messages = [{"role": "user", "content": content}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return messages

def count_input_tokens(provider: str, messages: list) -> int:
    # about how many tokens the messages are. text counts as 4 characters a token, images
    # by the rule the provider documents for their size in pixels
    characters = 0
    tokens = 0

    for message in messages:
        if isinstance(message["content"], str):
            characters += len(message["content"])
            continue

        for block in message["content"]:
            if block["type"] == "text":
                characters += len(block["text"])
            elif block["type"] == "image":
                tokens += _image_tokens(provider, base64.b64decode(block["source"]["data"]))
            elif block["type"] == "image_url":
                tokens += _image_tokens(provider, base64.b64decode(block["image_url"]["url"].split(",", 1)[1]))

    return characters // 4 + tokens

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
    # intra-case, one row per event
    "act_freqs": "share of every activity among the events of the case so far",
    "df_counts": "how often every directly-follows pair happened in the case so far",
    "act_set": "1 for every activity the case has already executed",
    "case_progress": "position of the event in its case, as a fraction of the case length",
    "current_act": "1 for the activity of the event itself",
    "past_acts": "1 for the activity of an earlier event, one block per step back",
    # resource, one row per calendar window
    "res_events": "events every resource executed in the window",
    "res_cases": "distinct cases every resource touched in the window",
    "res_durations": "mean event duration in minutes of every resource in the window",
    "res_waits": "mean minutes since the previous event of the case, over the events a resource took over from another one",
    "act_res_shares": "share of an activity's events in the window executed by a resource",
    "handover_shares": "share of a resource's handovers in the window that went to another resource",
    # inter-case, one row per calendar window
    "active_cases": "cases with at least one event in the window",
    "new_arrivals": "cases whose first event falls into the window",
    "completions": "cases whose last event falls into the window",
    "events_per_case": "events in the window divided by the active cases",
    "mean_delta_t": "mean minutes between an event and the previous event of its case",
    "std_delta_t": "standard deviation of those minutes",
    "stalled_cases": "running cases whose most recent event is older than the stall threshold at the window end",
    "attr_means": "mean of a numerical case attribute over the window's events",
    "attr_stds": "standard deviation of a numerical case attribute over the window's events",
    "attr_shares": "share of the window's events carrying a value of a categorical case attribute",
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
    # the intra-case features have one row per event, the resource and inter-case ones one
    # per calendar window
    values = features.drop(columns=META, errors="ignore")
    groups = values.columns.str.split(":").str[0]
    times = features["time:timestamp"]

    if "case:concept:name" in features.columns:
        lines = [
            f"Intra-case feature matrix: {len(features):,} rows, one per event, each describing "
            f"its case up to and including that event, with {values.shape[1]} feature columns.",
            f"The events belong to {features['case:concept:name'].nunique():,} cases and happened "
            f"from {times.min()} to {times.max()}.",
        ]
    else:
        perspective = "Resource" if set(groups) & set(RESOURCE_FEATURES) else "Inter-case"
        lines = [
            f"{perspective} feature matrix: {len(features):,} rows, one per calendar window of "
            f"{times.diff().min()}, with {values.shape[1]} feature columns.",
            f"The windows start from {times.min()} to {times.max()}.",
        ]

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

def abstract_states(frequencies: pd.Series, distances: pd.DataFrame, color_mapping: dict, clustering_scores: list[dict] | None = None) -> str:
    # frequencies is what get_som_state_frequencies, get_kmeans_state_frequencies or
    # get_dbscan_state_frequencies return, named after what they count, events or windows.
    # clustering_scores is a list of {"type": name, "value": score}, any scores in any order
    unit = frequencies.name
    total = frequencies.sum()
    never = [state for state in distances.index if state not in frequencies.index]

    lines = [f"{len(frequencies)} states occur among the {total:,} {unit}. {unit.capitalize()} per state:"]
    lines += [
        f"- {state}: {count:,} {unit} ({count / total:.1%})"
        for state, count in frequencies.sort_values(ascending=False).items()
    ]

    if never:
        lines.append(f"States without {unit}: " + ", ".join(never))

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

    if clustering_scores:
        lines.append("Clustering scores of the states:")
        lines += [f"- {score['type']}: {_format_score(score['value'])}" for score in clustering_scores]

    return "\n".join(lines)

def _format_score(value) -> str:
    # numbers get three decimals, anything else is written as it is
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "undefined" if pd.isna(value) else f"{value:,.3f}"

    return str(value)

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
        f"to {distributions.index[-1]}: the share of every state in each window.",
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

def abstract_log_trajectory(trajectory: pd.DataFrame) -> str:
    # trajectory is what get_som_log_trajectory, get_kmeans_log_trajectory or
    # get_dbscan_log_trajectory return, the resource and inter-case states window by window
    states = trajectory.groupby("state")
    windows = states["windows"].sum().sort_values(ascending=False)
    total = windows.sum()
    first, last = states["start"].min(), states["end"].max()
    moves = (trajectory["state"] + " -> " + trajectory["state"].shift(-1)).dropna().value_counts()

    lines = [
        f"Trajectory of the log through the states from {trajectory['start'].iloc[0]} to {trajectory['end'].iloc[-1]}: "
        f"{total:,} calendar windows in {len(trajectory):,} visits, a visit being a run of consecutive windows "
        "in the same state.",
        "Windows per state, and when the state first and last occurs:",
    ]
    lines += [
        f"- {state}: {count:,} windows ({count / total:.1%}), from {first[state]} to {last[state]}"
        for state, count in windows.items()
    ]

    lines.append("The most common moves from one state to the next:")
    lines += [f"- {move}: {count:,} times" for move, count in moves.head(10).items()]

    # hundreds of short visits do not fit into a prompt, the long ones show the phases of the log
    if len(trajectory) > 200:
        lines.append(f"The 200 longest of the {len(trajectory):,} visits, in order:")
        trajectory = trajectory.nlargest(200, "windows").sort_index()
    else:
        lines.append("The visits in order:")

    lines += [
        f"- {visit.state} from {visit.start} to {visit.end} ({visit.windows} windows)"
        for visit in trajectory.itertuples()
    ]

    return "\n".join(lines)

def abstract_divergences(divergences: pd.DataFrame, divergence: str, reference: str, lookback: int = 5) -> str:
    # divergences is what compute_divergences returns for this divergence, reference and lookback
    measure = f"the {DIVERGENCES[divergence]} between the state distribution of every calendar window"

    return _abstract_scores(divergences, measure, reference, lookback)

def abstract_window_distances(distances: pd.DataFrame, distance: str, reference: str, lookback: int = 5) -> str:
    # distances is what compute_window_distances returns for this distance, reference and lookback
    measure = f"the {DISTANCES[distance]} between the vector of every calendar window"

    return _abstract_scores(distances, measure, reference, lookback)

def _abstract_scores(scores: pd.DataFrame, measure: str, reference: str, lookback: int) -> str:
    # a score for every window, as compute_divergences and compute_window_distances give them
    against = f"the mean of the {lookback} windows before it" if reference == "recent" else f"the {REFERENCES[reference]}"
    values = scores["score"].dropna()

    lines = [
        f"Drift signal: {measure} and {against}, over {len(scores)} windows from {scores['window'].iloc[0]} "
        f"to {scores['window'].iloc[-1]}. The windows are numbered from 0 as in the plot.",
        f"Scores: median {values.median():.3f}, mean {values.mean():.3f}, highest {values.max():.3f}.",
        "The windows with the highest scores:",
    ]
    lines += [
        f"- window {number}, starting {row['window']}: {row['score']:.3f}"
        for number, row in scores.dropna().nlargest(5, "score").iterrows()
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
