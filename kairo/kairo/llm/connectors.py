"""LLM connectors: Anthropic, OpenAI, Google, and local OpenAI-compatible servers.

Every connector is a plain function `(prompt, ...) -> str`. API keys are read
from the environment at call time (never at import), checking the KAIRO_*
variable first and the provider's usual one second. Anthropic speaks its own
wire format; OpenAI, Google (via its OpenAI-compatible endpoint), and local
servers such as LM Studio or Ollama all share the chat-completions format.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

PROVIDERS = ("anthropic", "openai", "google", "local")

DEFAULTS: dict[str, dict] = {
    "anthropic": {
        "model": "claude-opus-5",
        "api_url": "https://api.anthropic.com/v1",
        "key_env": ("KAIRO_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    },
    "openai": {
        "model": "gpt-5",
        "api_url": "https://api.openai.com/v1",
        "key_env": ("KAIRO_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "tokens_key": "max_completion_tokens",
    },
    "google": {
        "model": "gemini-2.5-flash",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": ("KAIRO_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
        "tokens_key": "max_tokens",
    },
    "local": {
        "model": None,  # auto-discovered from the server's /models endpoint
        "api_url": "http://localhost:1234/v1",
        "key_env": ("KAIRO_LOCAL_API_KEY",),
        "url_env": "KAIRO_LOCAL_API_URL",
        "tokens_key": "max_tokens",
    },
}


def _env_key(provider: str) -> str | None:
    for name in DEFAULTS[provider]["key_env"]:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _image_b64(image_path: str | Path) -> tuple[str, str]:
    """The base64 payload and media type of an image file."""
    path = Path(image_path)
    suffix = path.suffix.lstrip(".").lower() or "png"
    media = f"image/{'jpeg' if suffix == 'jpg' else suffix}"
    return base64.b64encode(path.read_bytes()).decode("ascii"), media


def _raise_on_error(response: requests.Response) -> dict:
    payload = {}
    try:
        payload = response.json()
    except Exception:
        pass
    if not response.ok or "error" in payload:
        error = payload.get("error", {})
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"LLM request failed ({response.status_code}): {message or response.text[:300]}")
    return payload


def anthropic_query(
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    system: str | None = None,
    max_tokens: int = 16000,
    image_path: str | Path | None = None,
    extra_payload: dict | None = None,
    timeout: int = 600,
) -> str:
    """Execute a prompt against the Anthropic Messages API and return the text."""
    api_key = api_key or _env_key("anthropic")
    if not api_key:
        raise RuntimeError("No Anthropic API key — set KAIRO_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY")
    url = (api_url or DEFAULTS["anthropic"]["api_url"]).rstrip("/") + "/messages"
    content: object = prompt
    if image_path is not None:
        data, media = _image_b64(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
        ]
    payload: dict = {
        "model": model or DEFAULTS["anthropic"]["model"],
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if system:
        payload["system"] = system
    if extra_payload:
        payload.update(extra_payload)
    response = requests.post(
        url,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    data = _raise_on_error(response)
    # Thinking models return [thinking, text, ...] blocks — join the text ones.
    texts = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
    return "\n".join(texts).strip()


def _chat_completions_query(
    provider: str,
    prompt: str,
    model: str | None,
    api_key: str | None,
    api_url: str | None,
    system: str | None,
    max_tokens: int,
    image_path: str | Path | None,
    extra_payload: dict | None,
    timeout: int,
) -> str:
    """Shared transport for every OpenAI-compatible provider."""
    spec = DEFAULTS[provider]
    base = (api_url or os.environ.get(spec.get("url_env", ""), "") or spec["api_url"]).rstrip("/")
    api_key = api_key or _env_key(provider)
    if provider != "local" and not api_key:
        raise RuntimeError(f"No {provider} API key — set {' or '.join(spec['key_env'])}")
    model = model or spec["model"] or _first_model(base, api_key)
    content: object = prompt
    if image_path is not None:
        data, media = _image_b64(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}},
        ]
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": content})
    payload: dict = {"model": model, "messages": messages, spec["tokens_key"]: max_tokens}
    if extra_payload:
        payload.update(extra_payload)
    response = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key or 'local'}",
                 "content-type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    data = _raise_on_error(response)
    return (data["choices"][0]["message"]["content"] or "").strip()


def openai_query(
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    system: str | None = None,
    max_tokens: int = 16000,
    image_path: str | Path | None = None,
    extra_payload: dict | None = None,
    timeout: int = 600,
) -> str:
    """Execute a prompt against the OpenAI chat-completions API."""
    return _chat_completions_query("openai", prompt, model, api_key, api_url, system,
                                   max_tokens, image_path, extra_payload, timeout)


def google_query(
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    system: str | None = None,
    max_tokens: int = 16000,
    image_path: str | Path | None = None,
    extra_payload: dict | None = None,
    timeout: int = 600,
) -> str:
    """Execute a prompt against Gemini through Google's OpenAI-compatible endpoint."""
    return _chat_completions_query("google", prompt, model, api_key, api_url, system,
                                   max_tokens, image_path, extra_payload, timeout)


def local_query(
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    system: str | None = None,
    max_tokens: int = 16000,
    image_path: str | Path | None = None,
    extra_payload: dict | None = None,
    timeout: int = 600,
) -> str:
    """Execute a prompt against a local OpenAI-compatible server (LM Studio, Ollama…).

    The default URL is LM Studio's http://localhost:1234/v1 (override with
    `api_url` or KAIRO_LOCAL_API_URL). With `model=None` the first model the
    server reports as loaded is used.
    """
    return _chat_completions_query("local", prompt, model, api_key, api_url, system,
                                   max_tokens, image_path, extra_payload, timeout)


def available_models(provider: str = "local", api_url: str | None = None) -> list[str]:
    """The model ids an OpenAI-compatible server offers (GET /models)."""
    spec = DEFAULTS[provider]
    base = (api_url or os.environ.get(spec.get("url_env", ""), "") or spec["api_url"]).rstrip("/")
    response = requests.get(
        f"{base}/models",
        headers={"Authorization": f"Bearer {_env_key(provider) or 'local'}"},
        timeout=30,
    )
    data = _raise_on_error(response)
    return [entry["id"] for entry in data.get("data", [])]


def _first_model(base: str, api_key: str | None) -> str:
    response = requests.get(
        f"{base}/models", headers={"Authorization": f"Bearer {api_key or 'local'}"}, timeout=30
    )
    data = _raise_on_error(response)
    models = [entry["id"] for entry in data.get("data", [])]
    if not models:
        raise RuntimeError(f"No model loaded on {base} — load one (e.g. in LM Studio) or pass `model=`")
    return models[0]


QUERIES = {
    "anthropic": anthropic_query,
    "openai": openai_query,
    "google": google_query,
    "local": local_query,
}


def query(prompt: str, provider: str | None = None, **kwargs) -> str:
    """Execute a prompt against the chosen provider.

    The provider comes from the argument, then the KAIRO_LLM_PROVIDER
    environment variable, then defaults to "anthropic".
    """
    provider = provider or os.environ.get("KAIRO_LLM_PROVIDER", "anthropic")
    if provider not in QUERIES:
        raise ValueError(f"Unknown provider: {provider!r} (use one of {PROVIDERS})")
    return QUERIES[provider](prompt, **kwargs)
