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
        "Which range should be analyzed in detail for an intra-case drift?",
        "Which cases should be analyzed in detail for an intra-case drift?",
        "Which pipeline configuration and hyperparameters can I use to further inspect the drift signals?",
        "Is there an intra-case drift, and is it sudden, gradual or recurring?",
        "Which states change most around the strongest drift signal, and what behaviour do they stand for?",
        "What do the states stand for, in terms of activities and case progress?",
        "Are the states well separated, or would more or fewer states fit the log better?",
    ],
    "resource": [
        "Which range should be analyzed in detail for a resource drift?",
        "Which resources should be analyzed in detail for a resource drift?",
        "Which pipeline configuration and hyperparameters can I use to further inspect the drift signals?",
        "Is there a resource drift, and is it sudden, gradual or recurring?",
        "What do the states stand for, in terms of workload, waits and handovers of the resources?",
        "Are the states well separated, or would more or fewer states fit the log better?",
    ],
    "inter": [
        "Which range should be analyzed in detail for an inter-case drift?",
        "Which pipeline configuration and hyperparameters can I use to further inspect the drift signals?",
        "Is there an inter-case drift, and is it sudden, gradual or recurring?",
        "What do the states stand for: normal flow, congestion, bursts or slow phases?",
        "Which features change most around the strongest drift signal?",
        "Are the states well separated, or would more or fewer states fit the log better?",
    ],
}


def reset_system_prompt() -> None:
    st.session_state["copilot_sel_system_prompt"] = kairo.DEFAULT_SYSTEM_PROMPT


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
            state[f"{p}_frequencies"], state[f"{p}_distances"], state[f"{p}_colors"])

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


def show(p: str) -> None:
    ui.keep_widgets()
    log = ui.perspective_log(p)
    load_dotenv(find_dotenv())

    # the llm settings are the same for every perspective
    seed_widget("copilot_sel_provider", "anthropic")
    seed_widget("copilot_sel_host", "http://127.0.0.1:1234/v1")
    seed_widget("copilot_sel_api_key", "")
    seed_widget("copilot_sel_model", "claude-sonnet-5")
    seed_widget("copilot_sel_max_tokens", 64000)
    seed_widget("copilot_sel_system_prompt", kairo.DEFAULT_SYSTEM_PROMPT)
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
        st.text_area("System prompt", key="copilot_sel_system_prompt", height=320,
                     help="What the llm is told about kairo and the approach before every question.")
        st.button("Reset system prompt", on_click=reset_system_prompt, icon=":material/restart_alt:")

    st.title("Copilot")
    st.caption("Ask about the results so far. The llm sees only what is picked here.")

    abstractions = available_abstractions(p, log)
    plots = ui.stored_plots(p)

    seed_multi(f"{p}_sel_texts", list(abstractions), list(abstractions))
    seed_multi(f"{p}_sel_plots", [], list(plots))

    st.multiselect("Information to share", list(abstractions), key=f"{p}_sel_texts")
    st.multiselect("Plots to share", list(plots), key=f"{p}_sel_plots")

    # the summaries do not all say which perspective they come from
    picked = [abstractions[name]() for name in st.session_state[f"{p}_sel_texts"]]
    context = "\n\n".join([f"These results come from the {ui.PERSPECTIVES[p].lower()} perspective."] + picked) if picked else ""
    with st.expander("The text that is shared"):
        st.text(context or "Nothing picked.")

    st.pills("Suggestions", SUGGESTIONS[p], key=f"{p}_suggestion", on_change=use_suggestion, args=(p,))
    st.text_area("Question", key=f"{p}_sel_question", height=120)

    if st.button("Ask", type="primary", icon=":material/send:"):
        question = st.session_state[f"{p}_sel_question"].strip()
        if not question:
            st.warning("Enter a question first.")
            st.stop()

        connector = kairo.LLMConnector(
            provider=provider,
            model=st.session_state["copilot_sel_model"],
            api_key=st.session_state["copilot_sel_api_key"] or os.environ.get(KEY_VARIABLES.get(provider, ""), "EMPTY"),
            base_url=st.session_state["copilot_sel_host"] if provider == "custom" else None,
        )

        with st.spinner("Asking…"):
            try:
                input_tokens, messages = kairo.count_input_tokens(
                    provider,
                    st.session_state["copilot_sel_system_prompt"],
                    [context, f"Question: {question}"] if context else [question],
                    [plots[name] for name in st.session_state[f"{p}_sel_plots"]],
                )
                response = connector.call(messages=messages, max_tokens=int(st.session_state["copilot_sel_max_tokens"]))
            except Exception as exc:
                st.error(f"The request failed: {exc}")
                st.stop()

        st.session_state[f"{p}_answer"] = kairo.get_response_text(response)
        st.session_state[f"{p}_answer_tokens"] = (input_tokens, kairo.count_output_tokens(provider, response))

    if st.session_state.get(f"{p}_answer"):
        st.subheader("Answer")
        st.markdown(st.session_state[f"{p}_answer"])

    if f"{p}_answer_tokens" in st.session_state:
        input_tokens, output_tokens = st.session_state[f"{p}_answer_tokens"]
        st.caption(f"About {input_tokens:,} input tokens, estimated before sending · {output_tokens:,} output tokens, "
                   f"reasoning included, of at most {st.session_state['copilot_sel_max_tokens']:,}")
