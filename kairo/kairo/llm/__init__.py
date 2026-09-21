import base64
import plotly.graph_objects as go
from openai import OpenAI
from anthropic import Anthropic

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