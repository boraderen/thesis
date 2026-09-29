"""Copilot: ask an llm about the results of a perspective's pipeline so far."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

import kairo
import ui
from controls import seed_multi, seed_widget

PROVIDERS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "custom": "Custom host (OpenAI compatible)",
}
KEY_VARIABLES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}

SUGGESTIONS = {
    "intra": [
        "Is there an intra-case drift, when does it happen, and is it sudden, gradual or recurring?",
        "Which states change most around the strongest drift signal, and what case behaviour do they stand for?",
        "Which date range, cases and parameters should I inspect next to confirm the drift?",
    ],
    "resource": [
        "Is there a resource drift, when does it happen, and is it sudden, gradual or recurring?",
        "Which states change most around the strongest drift signal, and what resource behaviour do they stand for?",
        "Which date range, resources and parameters should I inspect next to confirm the drift?",
    ],
    "inter": [
        "Is there an inter-case drift, when does it happen, and is it sudden, gradual or recurring?",
        "Which states and features change most around the strongest drift signal, and what do they stand for?",
        "Which date range and parameters should I inspect next to confirm the drift?",
    ],
}


def use_suggestion(p: str) -> None:
    # the picked suggestion becomes the question, and the pills are cleared for the next one
    if st.session_state[f"{p}_suggestion"]:
        st.session_state[f"{p}_sel_question"] = st.session_state[f"{p}_suggestion"]
    st.session_state[f"{p}_suggestion"] = None


def available_abstractions(p: str, log: pd.DataFrame) -> dict:
    """Every step that ran, as a function turning its results into text, run only when picked."""
    state = st.session_state
    texts = {"Log statistics": lambda: kairo.abstract_log_stats(kairo.compute_log_stats(log))}

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


def new_chat(p: str) -> None:
    # the history goes, and the next question starts a new chat with nothing picked
    for key in ("chat_history", "chat_turns", "chat_provider", "sel_texts", "sel_plots"):
        st.session_state.pop(f"{p}_{key}", None)


def show_chat(p: str) -> None:
    # every turn as a question with what was shared along, and the answer with its tokens
    for turn in st.session_state.get(f"{p}_chat_turns", []):
        with st.chat_message("user"):
            st.markdown(turn["question"])
            shared = turn["texts"] + turn["plots"]
            st.caption("Shared: " + ", ".join(shared) if shared else "Nothing new shared")

        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            input_tokens, output_tokens = turn["tokens"]
            st.caption(f"About {input_tokens:,} input tokens, estimated before sending · {output_tokens:,} output tokens, "
                       f"reasoning included")


def show(p: str) -> None:
    ui.keep_widgets()
    log = ui.perspective_log(p)
    load_dotenv(find_dotenv())

    # after a question is sent, the box is emptied and nothing is shared again by default,
    # the llm still has it in the history
    if st.session_state.pop(f"{p}_chat_sent", False):
        st.session_state[f"{p}_sel_question"] = ""
        st.session_state[f"{p}_sel_texts"] = []
        st.session_state[f"{p}_sel_plots"] = []

    # the llm settings are the same for every perspective
    seed_widget("copilot_sel_provider", "anthropic")
    seed_widget("copilot_sel_host", "http://127.0.0.1:1234/v1")
    seed_widget("copilot_sel_api_key", "")
    seed_widget("copilot_sel_model", "claude-sonnet-5")
    seed_widget("copilot_sel_max_tokens", 64000)
    seed_widget(f"{p}_sel_question", "")

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
    st.caption("Ask about the results so far. The llm sees only what is picked here, and it keeps the chat, "
               "so what was shared once need not be shared again.")

    history = st.session_state.get(f"{p}_chat_history", [])
    show_chat(p)

    if history:
        st.button("New chat", on_click=new_chat, args=(p,), icon=":material/add_comment:")

    # the images are in the format of the provider the chat began with
    if history and st.session_state.get(f"{p}_chat_provider") != provider:
        st.warning(f"This chat is held with {PROVIDERS[st.session_state[f'{p}_chat_provider']]}. "
                   "Asking with another provider starts a new chat.")

    abstractions = available_abstractions(p, log)
    plots = ui.stored_plots(p)

    seed_multi(f"{p}_sel_texts", [], list(abstractions))
    seed_multi(f"{p}_sel_plots", [], list(plots))

    st.multiselect("Information to share", list(abstractions), key=f"{p}_sel_texts")
    st.multiselect("Plots to share", list(plots), key=f"{p}_sel_plots")

    # the summaries do not all say which perspective they come from
    picked = [abstractions[name]() for name in st.session_state[f"{p}_sel_texts"]]
    context = "\n\n".join([f"These results come from the {ui.PERSPECTIVES[p].lower()} perspective."] + picked) if picked else ""

    st.pills("Suggestions", SUGGESTIONS[p], key=f"{p}_suggestion", on_change=use_suggestion, args=(p,))
    st.text_area("Question", key=f"{p}_sel_question", height=120)

    if st.button("Ask", type="primary", icon=":material/send:"):
        question = st.session_state[f"{p}_sel_question"].strip()
        if not question:
            st.warning("Enter a question first.")
            st.stop()

        if history and st.session_state.get(f"{p}_chat_provider") != provider:
            st.session_state.pop(f"{p}_chat_turns", None)
            history = []

        connector = kairo.LLMConnector(
            provider=provider,
            model=st.session_state["copilot_sel_model"],
            api_key=st.session_state["copilot_sel_api_key"] or os.environ.get(KEY_VARIABLES.get(provider, ""), "EMPTY"),
            base_url=st.session_state["copilot_sel_host"] if provider == "custom" else None,
        )

        with st.spinner("Asking…"):
            try:
                # kairo's default system prompt starts the chat, every later question only adds a user message
                chat_history = history + kairo.create_messages(
                    provider,
                    None if history else kairo.DEFAULT_SYSTEM_PROMPT,
                    [context, f"Question: {question}"] if context else [question],
                    [plots[name] for name in st.session_state[f"{p}_sel_plots"]],
                )
                input_tokens = kairo.count_input_tokens(provider, chat_history)
                response = connector.call(messages=chat_history, max_tokens=int(st.session_state["copilot_sel_max_tokens"]))
            except Exception as exc:
                st.error(f"The request failed: {exc}")
                st.stop()

        answer = kairo.get_response_text(response)
        chat_history.append({"role": "assistant", "content": answer})

        st.session_state[f"{p}_chat_history"] = chat_history
        st.session_state[f"{p}_chat_provider"] = provider
        st.session_state.setdefault(f"{p}_chat_turns", []).append({
            "question": question,
            "texts": list(st.session_state[f"{p}_sel_texts"]),
            "plots": list(st.session_state[f"{p}_sel_plots"]),
            "answer": answer,
            "tokens": (input_tokens, kairo.count_output_tokens(provider, response)),
        })
        st.session_state[f"{p}_chat_sent"] = True
        st.rerun()
