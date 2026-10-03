"""Copilot: ask an llm about the results of all three perspectives' pipelines so far."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

import kairo
import ui
from controls import seed_multi, seed_widget

PROVIDERS = {
    "custom": "Custom host (OpenAI compatible)",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
}
KEY_VARIABLES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}

SUGGESTIONS = [
    "Is there a drift, when does it happen, which perspectives show it, and is it sudden, gradual or recurring?",
    "Do the perspectives agree on the drift, and what does each of them say changed?",
    "Which states change most around the strongest drift signal, and what behaviour do they stand for?",
    "Which perspective, date range and parameters should I inspect next to confirm the drift?",
]


def use_suggestion() -> None:
    # the picked suggestion becomes the question, and the pills are cleared for the next one
    if st.session_state["copilot_suggestion"]:
        st.session_state["copilot_sel_question"] = st.session_state["copilot_suggestion"]
    st.session_state["copilot_suggestion"] = None


def perspective_abstractions(p: str) -> dict:
    """Every step of a perspective that ran, as a function turning its results into text, run only when picked."""
    state = st.session_state
    texts = {}

    if f"{p}_features" in state:
        texts["Features"] = lambda: kairo.abstract_features(state[f"{p}_features"])

    if f"{p}_cut" in state:
        texts["PCA"] = lambda: kairo.abstract_pca(state[f"{p}_pca"], state[f"{p}_cut"])

    if f"{p}_states" in state:
        texts["States"] = lambda: kairo.abstract_states(
            state[f"{p}_frequencies"], state[f"{p}_distances"], state[f"{p}_colors"],
            [{"type": name, "value": value} for name, value, _ in state.get(f"{p}_scores", [])])

    for case_id, trajectory in state.get(f"{p}_trajectories", {}).items():
        texts[f"Trajectory of case {case_id}"] = (
            lambda visits=trajectory["visits"], case_id=case_id: kairo.abstract_case_trajectory(visits, case_id))

    if f"{p}_range_trajectories" in state:
        texts["Trajectories of the cases in a range"] = lambda: kairo.abstract_case_trajectories(
            state[f"{p}_range_trajectories"])

    if f"{p}_log_trajectory" in state:
        texts["Trajectory of the log"] = lambda: kairo.abstract_log_trajectory(state[f"{p}_log_trajectory"])

    if f"{p}_distributions" in state:
        texts["State distributions"] = lambda: kairo.abstract_distributions(state[f"{p}_distributions"])

    if f"{p}_divergences" in state:
        texts["Drift signal"] = lambda: kairo.abstract_divergences(
            state[f"{p}_divergences"], **state[f"{p}_divergence_config"])

    if f"{p}_window_distances" in state:
        texts["Window distances"] = lambda: kairo.abstract_window_distances(
            state[f"{p}_window_distances"], **state[f"{p}_window_distance_config"])

    return texts


def available_abstractions(log: pd.DataFrame) -> dict:
    # the log statistics once, then the steps of every perspective under its name. the
    # summaries do not all say which perspective they come from, so the text says it
    texts = {"Log statistics": lambda: kairo.abstract_log_stats(kairo.compute_log_stats(log))}

    for p, label in ui.PERSPECTIVES.items():
        for name, text in perspective_abstractions(p).items():
            texts[f"{label} · {name}"] = lambda text=text, title=f"{label} perspective · {name}": f"{title}:\n{text()}"

    return texts


def available_plots() -> dict:
    return {f"{label} · {name}": plot
            for p, label in ui.PERSPECTIVES.items() for name, plot in ui.stored_plots(p).items()}


def new_chat() -> None:
    # the history goes, and the next question starts a new chat with nothing picked
    for key in ("chat_history", "chat_turns", "chat_provider", "sel_texts", "sel_plots"):
        st.session_state.pop(f"copilot_{key}", None)


def show_chat() -> None:
    # every turn as a question with what was shared along, and the answer with its tokens
    for turn in st.session_state.get("copilot_chat_turns", []):
        with st.chat_message("user"):
            st.markdown(turn["question"])
            shared = turn["texts"] + turn["plots"]
            st.caption("Shared: " + ", ".join(shared) if shared else "Nothing new shared")

        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            input_tokens, output_tokens = turn["tokens"]
            st.caption(f"About {input_tokens:,} input tokens, estimated before sending · {output_tokens:,} output tokens, "
                       f"reasoning included")


ui.keep_widgets()
log = ui.require_log()
# every perspective drops its results if they were computed on another log
for p in ui.PERSPECTIVES:
    ui.perspective_log(p)
load_dotenv(find_dotenv())

# after a question is sent, the box is emptied and nothing is shared again by default,
# the llm still has it in the history
if st.session_state.pop("copilot_chat_sent", False):
    st.session_state["copilot_sel_question"] = ""
    st.session_state["copilot_sel_texts"] = []
    st.session_state["copilot_sel_plots"] = []

seed_widget("copilot_sel_provider", "custom")
seed_widget("copilot_sel_host", "http://127.0.0.1:1234/v1")
seed_widget("copilot_sel_api_key", "")
seed_widget("copilot_sel_model", "Qwen/Qwen3-VL-32B-Instruct")
seed_widget("copilot_sel_max_tokens", 64000)
seed_widget("copilot_sel_question", "")

with st.sidebar:
    st.header("Copilot")
    provider = st.selectbox("Provider", list(PROVIDERS), format_func=PROVIDERS.get, key="copilot_sel_provider")
    if provider == "custom":
        st.text_input("Host", key="copilot_sel_host", help="Base URL of the server, e.g. LM Studio or vLLM.")
    st.text_input("API key", type="password", key="copilot_sel_api_key",
                  help=f"Left empty, {KEY_VARIABLES.get(provider, 'no key')} from the environment is used.")
    st.text_input("Model", key="copilot_sel_model")
    st.number_input("Max tokens", min_value=100, step=1000, key="copilot_sel_max_tokens")

st.title("Copilot")
st.caption("Ask about the results of every perspective so far. The llm sees only what is picked here, and it "
           "keeps the chat, so what was shared once need not be shared again.")

history = st.session_state.get("copilot_chat_history", [])
show_chat()

if history:
    st.button("New chat", on_click=new_chat, icon=":material/add_comment:")

# the images are in the format of the provider the chat began with
if history and st.session_state.get("copilot_chat_provider") != provider:
    st.warning(f"This chat is held with {PROVIDERS[st.session_state['copilot_chat_provider']]}. "
               "Asking with another provider starts a new chat.")

abstractions = available_abstractions(log)
plots = available_plots()

seed_multi("copilot_sel_texts", [], list(abstractions))
seed_multi("copilot_sel_plots", [], list(plots))

st.multiselect("Information to share", list(abstractions), key="copilot_sel_texts")
st.multiselect("Plots to share", list(plots), key="copilot_sel_plots")

st.pills("Suggestions", SUGGESTIONS, key="copilot_suggestion", on_change=use_suggestion)
st.text_area("Question", key="copilot_sel_question", height=120)

if st.button("Ask", type="primary", icon=":material/send:"):
    question = st.session_state["copilot_sel_question"].strip()
    if not question:
        st.warning("Enter a question first.")
        st.stop()

    if history and st.session_state.get("copilot_chat_provider") != provider:
        st.session_state.pop("copilot_chat_turns", None)
        history = []

    connector = kairo.LLMConnector(
        provider=provider,
        model=st.session_state["copilot_sel_model"],
        api_key=st.session_state["copilot_sel_api_key"] or os.environ.get(KEY_VARIABLES.get(provider, ""), "EMPTY"),
        base_url=st.session_state["copilot_sel_host"] if provider == "custom" else None,
    )

    with st.spinner("Asking…"):
        try:
            # the plots go along as bare images, so the text names them in the order they are attached
            picked_plots = st.session_state["copilot_sel_plots"]
            shared = [abstractions[name]() for name in st.session_state["copilot_sel_texts"]]
            if picked_plots:
                shared.append("The plots, in the order they are attached:\n"
                              + "\n".join(f"{i}. {name}" for i, name in enumerate(picked_plots, 1)))

            # kairo's default system prompt starts the chat, every later question only adds a user message
            chat_history = history + kairo.create_messages(
                provider,
                None if history else kairo.DEFAULT_SYSTEM_PROMPT,
                shared + [f"Question: {question}"] if shared else [question],
                [plots[name] for name in picked_plots],
            )
            input_tokens = kairo.count_input_tokens(provider, chat_history)
            response = connector.call(messages=chat_history, max_tokens=int(st.session_state["copilot_sel_max_tokens"]))
        except Exception as exc:
            st.error(f"The request failed: {exc}")
            st.stop()

    answer = kairo.get_response_text(response)
    chat_history.append({"role": "assistant", "content": answer})

    st.session_state["copilot_chat_history"] = chat_history
    st.session_state["copilot_chat_provider"] = provider
    st.session_state.setdefault("copilot_chat_turns", []).append({
        "question": question,
        "texts": list(st.session_state["copilot_sel_texts"]),
        "plots": list(picked_plots),
        "answer": answer,
        "tokens": (input_tokens, kairo.count_output_tokens(provider, response)),
    })
    st.session_state["copilot_chat_sent"] = True
    st.rerun()
