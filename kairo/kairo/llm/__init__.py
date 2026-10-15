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
from ..analysis.intra import GLOSSARY as INTRA_GLOSSARY
from ..analysis.inter import GLOSSARY as INTER_GLOSSARY
from ..analysis.resource import FEATURES as RESOURCE_FEATURES, GLOSSARY as RESOURCE_GLOSSARY

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
An event log has one row per event with a case id, an activity and a timestamp, and optionally a resource, durations and case attributes. Kairo describes the process through states and follows how often every state occurs over time. When the process changes, the mix of states changes, and comparing that mix between signal windows gives a drift signal. Kairo uses calendar windows, time intervals of fixed length counted from 1 January 1970, in two roles: a feature window (e.g. 1D) is one row of the resource and inter-case features, a signal window (e.g. 7D, at least as long as the feature windows) is one point of the drift signal. Three perspectives run the same pipeline:
- Intra-case: the situation of a single running case, one row per event.
- Resource: how the resources work, one row per feature window.
- Inter-case: the situation across all running cases, one row per feature window.
The pipeline: features, then optional scaling (z-score, or min-max to 0-1) and optional PCA (fit components, cut where the explained variance flattens), then clustering into states (SOM grid cells, k-means clusters, or DBSCAN dense regions with noise), then trajectories through the states, then the drift signal: the share of every state per signal window (the events, or for resource and inter-case the feature windows, that fall into it) compared to a reference (previous signal window, mean of the recent signal windows, or mean of all signal windows) by KL (unbounded, explodes when a state is missing on one side), Jensen-Shannon (at most ln 2), total variation (0-1) or Hellinger (0-1). The drift signal is a time series: for every signal window its divergence, the drift score. It is the only drift signal of every perspective. State colors follow the distances between the states: close states get similar colors, far states very different ones.

THE FEATURES
""" + "\n".join(
    f"{title}\n" + "\n".join(f"- {group} ({label}): {meaning}" for group, (label, meaning) in glossary.items())
    for title, glossary in (
        ("Intra-case, one row per event, describing its prefix, i.e., its case up to and including the event:", INTRA_GLOSSARY),
        ("Resource, one row per feature window:", RESOURCE_GLOSSARY),
        ("Inter-case, one row per feature window, over all cases:", INTER_GLOSSARY),
    )
) + """

WHAT KAIRO CAN PROVIDE
Log statistics; the features; PCA explained variances and the strongest features per component; states with their sizes, distances between them and colors; SOM u-matrix and heatmaps; DBSCAN k-distance curve for picking eps; state frequencies for any date range, e.g. before and after a suspected drift; the trajectory of a single case, of all cases in a date range, or of the log feature window by feature window; state distributions per signal window; the drift signal; and clustering scores: silhouette (-1 to 1, above about 0.5 is good, below 0.25 weak), Calinski-Harabasz (higher is better, only comparable on the same rows), DBCV for DBSCAN (-1 to 1, noise counts against it), SOM quantization error (lower is closer, but bigger grids always lower it) and SOM occupancy entropy (0 to 1, how evenly the grid is used). Everything can be rerun with other feature and signal windows, features, scaling, PCA cuts, methods, parameters, divergences, references and date ranges.

STRENGTHS AND LIMITS
- Intra-case features are pure control flow: they see new, missing or reordered activities, loops and skips, but no time, resources or attributes. A process that only got slower is invisible there.
- Resource features see who does what, workload, durations, waits and handovers, but not the order of activities within a case.
- Inter-case features see load, arrivals, pace, stalls and case mix, but not which activities or resources.
- So drifts do not always spread. A control-flow change shows in intra-case and often, but not always, in resource shares or events per case. A staffing or reassignment change shows in resource and maybe in inter-case pace, not in intra-case. A workload or arrival change shows in inter-case and resource volume, not in intra-case. A slowdown shows in inter-case and resource times only. Expect a perspective to react only when its features can see the change, and do not read a quiet perspective as evidence against a drift it cannot see.
- States are unsupervised and depend on every earlier choice. SOM cell numbers change with the grid size and the input rows, k-means favours round equal clusters, DBSCAN on intra-case prefixes (many identical rows) tends to find many states.
- Resource and inter-case have only as many rows as feature windows, often a few hundred, so their clusterings are fragile. Empty or quiet feature windows (weekends, holidays) can form their own state and look like a recurring drift.
- Signals at the very start or end of the log (truncated cases), in signal windows with few events, or from KL on a state missing in one signal window are often artefacts.
- The signal tells when the mix changed, not why. The why comes from which states changed and what they stand for.

HOW TO WORK
- Use only what the user shares: summaries and plots. Never invent numbers, states, dates, cases or resources, and name them as the summaries do ((i, j) for SOM cells, numbers for clusters, "noise"; feature and signal windows by start date).
- Read every detail and connect them: which states grow or shrink around a spike, what their features mean, whether the timing fits the event counts.
- Judge the configuration before the signal. Say plainly when it is not meaningful and why, e.g. one state holding almost all rows, many empty SOM cells (low occupancy entropy), weak silhouette, DBSCAN mostly noise or one giant cluster, a PCA cut keeping little variance or dominated by a few rare z-scored columns, signal windows too short for the event volume or holding only a few feature windows, features that cannot see the suspected change. Then suggest a concrete better configuration, for one or all perspectives, with values.
- Be skeptical of drift signals. Trust a drift when it is consistent: the same dates across divergences, references and signal windows, a clear shift in states that stand for real behaviour, and agreement between perspectives that can see it. When results contradict each other or could be artefacts, say so and name the one or two checks that would settle it (a date filter around the spike, frequencies before and after, a case or log trajectory, another feature or signal window, divergence, method or parameters). When the evidence is clear, state the drift directly: the date range, the type (sudden: one isolated spike; gradual: a stretch of raised scores; recurring: repeated spikes), the perspectives it shows in, and what changed.
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

# what every feature group computes, from the glossaries next to the feature functions
FEATURE_GROUPS = {
    group: meaning
    for glossary in (INTRA_GLOSSARY, RESOURCE_GLOSSARY, INTER_GLOSSARY)
    for group, (label, meaning) in glossary.items()
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
    # per feature window
    values = features.drop(columns=META, errors="ignore")
    groups = values.columns.str.split(":").str[0]
    times = features["time:timestamp"]

    if "case:concept:name" in features.columns:
        lines = [
            f"Intra-case feature matrix: {len(features):,} rows, one per event, each describing "
            f"its prefix, i.e., its case up to and including that event, with {values.shape[1]} feature columns.",
            f"The events belong to {features['case:concept:name'].nunique():,} cases and happened "
            f"from {times.min()} to {times.max()}.",
        ]
    else:
        perspective = "Resource" if set(groups) & set(RESOURCE_FEATURES) else "Inter-case"
        lines = [
            f"{perspective} feature matrix: {len(features):,} rows, one per feature window of "
            f"{times.diff().min()}, with {values.shape[1]} feature columns.",
            f"The feature windows start from {times.min()} to {times.max()}.",
        ]

    lines.append("Feature groups:")

    for group in dict.fromkeys(groups):
        block = values.loc[:, groups == group].to_numpy()
        means = values.loc[:, groups == group].mean().sort_values(ascending=False)
        meaning = next((text for name, text in FEATURE_GROUPS.items() if group.startswith(name)), "")

        lines.append(
            f"- {group}: {meaning} Here {block.shape[1]} columns, mean value {block.mean():.3f}, "
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

def abstract_states(frequencies: pd.Series, distances: pd.DataFrame, state_colors: dict, clustering_scores: list[dict] | None = None) -> str:
    # frequencies is what get_som_state_frequencies, get_kmeans_state_frequencies or
    # get_dbscan_state_frequencies return, named after what they count, events or feature windows.
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

    lines.append("Colors of the states in the plots, close states have similar colors:")

    for key, color in state_colors.items():
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
    # trajectories is what get_som_case_trajectories, get_kmeans_case_trajectories or
    # get_dbscan_case_trajectories return. hundreds of cases do not fit into a prompt one by one,
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

def abstract_state_distributions(distributions: pd.DataFrame) -> str:
    # distributions is what compute_state_distributions returns
    changes = distributions.diff()
    # half the summed changes of the shares, how much of the mix moved since the signal window before
    moved = changes.abs().sum(axis=1) / 2

    lines = [
        f"State distribution over {len(distributions)} signal windows from {distributions.index[0]} "
        f"to {distributions.index[-1]}: the share of every state in each signal window.",
        "Share of every state over the signal windows, mean, lowest and highest:",
    ]

    for state in distributions.columns:
        shares = distributions[state]
        lines.append(
            f"- {state}: mean {shares.mean():.1%}, lowest {shares.min():.1%} in the signal window starting "
            f"{shares.idxmin()}, highest {shares.max():.1%} in the signal window starting {shares.idxmax()}"
        )

    lines.append("The largest changes from one signal window to the next, with the states that changed most:")

    for signal_window, share in moved.nlargest(5).items():
        change = changes.loc[signal_window]
        lines.append(
            f"- signal window starting {signal_window}: {share:.1%} of the mix moved, "
            + ", ".join(f"{state} {change[state]:+.1%}" for state in change.abs().nlargest(3).index)
        )

    return "\n".join(lines)

def abstract_log_trajectory(trajectory: pd.DataFrame) -> str:
    # trajectory is what get_som_log_trajectory, get_kmeans_log_trajectory or
    # get_dbscan_log_trajectory return, the resource and inter-case states feature window by feature window
    states = trajectory.groupby("state")
    windows = states["feature_windows"].sum().sort_values(ascending=False)
    total = windows.sum()
    first, last = states["start"].min(), states["end"].max()
    moves = (trajectory["state"] + " -> " + trajectory["state"].shift(-1)).dropna().value_counts()

    lines = [
        f"Trajectory of the log through the states from {trajectory['start'].iloc[0]} to {trajectory['end'].iloc[-1]}: "
        f"{total:,} feature windows in {len(trajectory):,} visits, a visit being a run of consecutive feature "
        "windows in the same state.",
        "Feature windows per state, and when the state first and last occurs:",
    ]
    lines += [
        f"- {state}: {count:,} feature windows ({count / total:.1%}), from {first[state]} to {last[state]}"
        for state, count in windows.items()
    ]

    lines.append("The most common moves from one state to the next:")
    lines += [f"- {move}: {count:,} times" for move, count in moves.head(10).items()]

    # hundreds of short visits do not fit into a prompt, the long ones show the phases of the log
    if len(trajectory) > 200:
        lines.append(f"The 200 longest of the {len(trajectory):,} visits, in order:")
        trajectory = trajectory.nlargest(200, "feature_windows").sort_index()
    else:
        lines.append("The visits in order:")

    lines += [
        f"- {visit.state} from {visit.start} to {visit.end} ({visit.feature_windows} feature windows)"
        for visit in trajectory.itertuples()
    ]

    return "\n".join(lines)

def abstract_drift_signal(drift_signal: pd.DataFrame, divergence: str, reference: str, lookback: int = 5) -> str:
    # drift_signal is what compute_drift_signal returns for this divergence, reference and lookback
    against = (f"the mean of the {lookback} signal windows before it" if reference == "recent"
               else f"the {REFERENCES[reference]}")
    scores = drift_signal["drift_score"].dropna()

    lines = [
        f"Drift signal: for every signal window, the {DIVERGENCES[divergence]} between its state distribution "
        f"and {against}, the drift score of the signal window. {len(drift_signal)} signal windows from "
        f"{drift_signal['signal_window'].iloc[0]} to {drift_signal['signal_window'].iloc[-1]}.",
        f"Drift scores: median {scores.median():.3f}, mean {scores.mean():.3f}, highest {scores.max():.3f}.",
        "The signal windows with the highest drift scores:",
    ]
    lines += [
        f"- signal window starting {row['signal_window']}: {row['drift_score']:.3f}"
        for _, row in drift_signal.dropna().nlargest(5, "drift_score").iterrows()
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
