import base64
import colorsys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from anthropic import Anthropic
from sklearn.decomposition import PCA

from ..analysis import META

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

    def call(self, *args, **kwargs):
        return self.connector.call(*args, **kwargs)


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

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        plots: list[go.Figure] | None = None,
        max_tokens: int = 1000,
    ):
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        if plots:
            for fig in plots:
                image_bytes = fig.to_image(format="png")
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                })

        messages.append({
            "role": "user",
            "content": content,
        })

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

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        plots: list[go.Figure] | None = None,
        max_tokens: int = 1000,
    ):
        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        if plots:
            for fig in plots:
                image_bytes = fig.to_image(format="png")
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64,
                    },
                })

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        return self.client.messages.create(**kwargs)


def get_response_text(response) -> str:
    # the openai style apis answer with choices, anthropic with content blocks
    if hasattr(response, "choices"):
        return response.choices[0].message.content

    return "".join(block.text for block in response.content if block.type == "text")


# abstractions: every step of the pipeline turned into text an llm can read

FEATURE_GROUPS = {
    "act_freqs": "share of every activity among the events of the case so far",
    "df_counts": "how often every directly-follows pair happened in the case so far",
    "act_set": "1 for every activity the case has already executed",
    "case_progress": "position of the event in its case, as a fraction of the case length",
    "current_act": "1 for the activity of the event itself",
    "past_acts": "1 for the activity of an earlier event, one block per step back",
}

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

def abstract_states(states: pd.DataFrame, distances: pd.DataFrame, color_mapping: dict) -> str:
    # states has per event either its som cell (i, j) or its cluster
    if "cluster" in states.columns:
        names = states["cluster"].map(lambda c: "noise" if c == -1 else str(c))
    else:
        names = "(" + states["i"].astype(str) + ", " + states["j"].astype(str) + ")"

    counts = names.value_counts()
    never = [state for state in distances.index if state not in counts.index]

    lines = [f"{len(counts)} states occur among the {len(states):,} events. Events per state:"]
    lines += [f"- {state}: {count:,} events ({count / len(states):.1%})" for state, count in counts.items()]

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

def _color_name(color: str) -> str:
    # a rough name, so the llm can match "the purple bar" in a plot to its state
    r, g, b = (int(color[k:k + 2], 16) / 255 for k in (1, 3, 5))
    hue, saturation, _ = colorsys.rgb_to_hsv(r, g, b)

    if saturation < 0.2:
        return "grey"

    hues = [(0.04, "red"), (0.11, "orange"), (0.19, "yellow"), (0.45, "green"), (0.54, "cyan"),
            (0.72, "blue"), (0.83, "purple"), (0.95, "pink"), (1.0, "red")]

    return next(name for bound, name in hues if hue <= bound)
