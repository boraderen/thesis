"""Copilot: ask an llm about the results of the intra-case pipeline so far."""
from __future__ import annotations

import os

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

SUGGESTIONS = [
    "Which range should be analyzed in detail for an intra-case drift?",
    "Which cases should be analyzed in detail for an intra-case drift?",
    "Which pipeline configuration and hyperparameters can I use to further inspect the drift signals?",
    "Is there an intra-case drift, and is it sudden, gradual or recurring?",
    "Which states change most around the strongest drift signal, and what behaviour do they stand for?",
    "What do the states stand for, in terms of activities and case progress?",
    "Are the states well separated, or would more or fewer states fit the log better?",
]

ui.keep_widgets()
log = ui.intra_log()
load_dotenv(find_dotenv())

seed_widget("intra_sel_provider", "anthropic")
seed_widget("intra_sel_host", "http://127.0.0.1:1234/v1")
seed_widget("intra_sel_api_key", "")
seed_widget("intra_sel_model", "claude-sonnet-5")
seed_widget("intra_sel_max_tokens", 2000)
seed_widget("intra_sel_question", "")
seed_widget("intra_sel_system_prompt", kairo.DEFAULT_SYSTEM_PROMPT)


def reset_system_prompt() -> None:
    st.session_state["intra_sel_system_prompt"] = kairo.DEFAULT_SYSTEM_PROMPT


def use_suggestion() -> None:
    # the picked suggestion becomes the question, and the pills are cleared for the next one
    if st.session_state["intra_suggestion"]:
        st.session_state["intra_sel_question"] = st.session_state["intra_suggestion"]
    st.session_state["intra_suggestion"] = None


with st.sidebar:
    st.header("Copilot")
    provider = st.selectbox("Provider", list(PROVIDERS), format_func=PROVIDERS.get, key="intra_sel_provider")
    if provider == "custom":
        st.text_input("Host", key="intra_sel_host", help="Base URL of the server, e.g. LM Studio or vLLM.")
    st.text_input("API key", type="password", key="intra_sel_api_key",
                  help=f"Left empty, {KEY_VARIABLES.get(provider, 'no key')} from the environment is used.")
    st.text_input("Model", key="intra_sel_model")
    st.number_input("Max tokens", min_value=100, max_value=64000, step=100, key="intra_sel_max_tokens")
    st.text_area("System prompt", key="intra_sel_system_prompt", height=320,
                 help="What the llm is told about kairo and the approach before every question.")
    st.button("Reset system prompt", on_click=reset_system_prompt, icon=":material/restart_alt:")


def available_abstractions() -> dict:
    """Every step that ran, as a function turning its results into text, run only when picked."""
    texts = {"Log statistics": lambda: kairo.abstract_log_stats(kairo.compute_log_stats(log))}

    if "intra_features" in st.session_state:
        texts["Features"] = lambda: kairo.abstract_features(st.session_state["intra_features"])

    if "intra_cut" in st.session_state:
        texts["PCA"] = lambda: kairo.abstract_pca(st.session_state["intra_pca"], st.session_state["intra_cut"])

    if "intra_states" in st.session_state:
        texts["States"] = lambda: kairo.abstract_states(
            st.session_state["intra_frequencies"], st.session_state["intra_distances"], st.session_state["intra_colors"])

    for case_id, trajectory in st.session_state.get("intra_trajectories", {}).items():
        texts[f"Trajectory of case {case_id}"] = (
            lambda visits=trajectory["visits"], case_id=case_id: kairo.abstract_case_trajectory(visits, case_id))

    if "intra_range_trajectories" in st.session_state:
        texts["Trajectories of the cases in a range"] = lambda: kairo.abstract_case_trajectories(
            st.session_state["intra_range_trajectories"])

    if "intra_distributions" in st.session_state:
        texts["State distributions"] = lambda: kairo.abstract_distributions(st.session_state["intra_distributions"])

    if "intra_divergences" in st.session_state:
        texts["Drift signal"] = lambda: kairo.abstract_divergences(
            st.session_state["intra_divergences"], **st.session_state["intra_divergence_config"])

    return texts


st.title("Copilot")
st.caption("Ask about the results so far. The llm sees only what is picked here.")

abstractions = available_abstractions()
plots = ui.stored_plots()

if not abstractions and not plots:
    st.info("Nothing to share yet — run the steps of the pipeline first.")
    st.stop()

seed_multi("intra_sel_texts", list(abstractions), list(abstractions))
seed_multi("intra_sel_plots", [], list(plots))

st.multiselect("Information to share", list(abstractions), key="intra_sel_texts")
st.multiselect("Plots to share", list(plots), key="intra_sel_plots")

context = "\n\n".join(abstractions[name]() for name in st.session_state["intra_sel_texts"])
with st.expander("The text that is shared"):
    st.text(context or "Nothing picked.")

st.pills("Suggestions", SUGGESTIONS, key="intra_suggestion", on_change=use_suggestion)
st.text_area("Question", key="intra_sel_question", height=120)

if st.button("Ask", type="primary", icon=":material/send:"):
    question = st.session_state["intra_sel_question"].strip()
    if not question:
        st.warning("Enter a question first.")
        st.stop()

    connector = kairo.LLMConnector(
        provider=provider,
        model=st.session_state["intra_sel_model"],
        api_key=st.session_state["intra_sel_api_key"] or os.environ.get(KEY_VARIABLES.get(provider, ""), "EMPTY"),
        base_url=st.session_state["intra_sel_host"] if provider == "custom" else None,
    )

    with st.spinner("Asking…"):
        try:
            input_tokens, messages = kairo.count_input_tokens(
                provider,
                st.session_state["intra_sel_system_prompt"],
                [context, f"Question: {question}"] if context else [question],
                [plots[name] for name in st.session_state["intra_sel_plots"]],
            )
            response = connector.call(messages=messages, max_tokens=int(st.session_state["intra_sel_max_tokens"]))
        except Exception as exc:
            st.error(f"The request failed: {exc}")
            st.stop()

    st.session_state["intra_answer"] = kairo.get_response_text(response)
    st.session_state["intra_answer_tokens"] = (input_tokens, kairo.count_output_tokens(provider, response))

if st.session_state.get("intra_answer"):
    st.subheader("Answer")
    st.markdown(st.session_state["intra_answer"])

if "intra_answer_tokens" in st.session_state:
    input_tokens, output_tokens = st.session_state["intra_answer_tokens"]
    st.caption(f"About {input_tokens:,} input tokens, estimated before sending · {output_tokens:,} output tokens, "
               f"reasoning included, of at most {st.session_state['intra_sel_max_tokens']:,}")
