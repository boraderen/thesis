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

SYSTEM_PROMPT = (
    "You help to analyse an event log with state-based process monitoring. Every event got a "
    "feature vector describing its case up to that event, PCA compressed these vectors, and a "
    "clustering (SOM, k-means or DBSCAN) turned them into states. The user shares text summaries "
    "of these steps and may attach plots. Answer from what is shared, and say so when it is not "
    "enough to answer."
)

ui.keep_widgets()
ui.intra_log()
load_dotenv(find_dotenv())

seed_widget("intra_sel_provider", "anthropic")
seed_widget("intra_sel_host", "http://127.0.0.1:1234/v1")
seed_widget("intra_sel_api_key", "")
seed_widget("intra_sel_model", "claude-sonnet-5")
seed_widget("intra_sel_max_tokens", 2000)
seed_widget("intra_sel_question", "")

with st.sidebar:
    st.header("Copilot")
    provider = st.selectbox("Provider", list(PROVIDERS), format_func=PROVIDERS.get, key="intra_sel_provider")
    if provider == "custom":
        st.text_input("Host", key="intra_sel_host", help="Base URL of the server, e.g. LM Studio or vLLM.")
    st.text_input("API key", type="password", key="intra_sel_api_key",
                  help=f"Left empty, {KEY_VARIABLES.get(provider, 'no key')} from the environment is used.")
    st.text_input("Model", key="intra_sel_model")
    st.number_input("Max tokens", min_value=100, max_value=64000, step=100, key="intra_sel_max_tokens")


def available_abstractions() -> dict:
    """Every step that ran, as a function turning its results into text, run only when picked."""
    texts = {}

    if "intra_features" in st.session_state:
        texts["Features"] = lambda: kairo.abstract_features(st.session_state["intra_features"])

    if "intra_cut" in st.session_state:
        texts["PCA"] = lambda: kairo.abstract_pca(st.session_state["intra_pca"], st.session_state["intra_cut"])

    if "intra_states" in st.session_state:
        texts["States"] = lambda: kairo.abstract_states(
            st.session_state["intra_states"], st.session_state["intra_distances"], st.session_state["intra_colors"])

    for case_id, trajectory in st.session_state.get("intra_trajectories", {}).items():
        texts[f"Trajectory of case {case_id}"] = (
            lambda visits=trajectory["visits"], case_id=case_id: kairo.abstract_case_trajectory(visits, case_id))

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
            response = connector.call(
                prompt=f"{context}\n\nQuestion: {question}" if context else question,
                system_prompt=SYSTEM_PROMPT,
                plots=[plots[name] for name in st.session_state["intra_sel_plots"]],
                max_tokens=int(st.session_state["intra_sel_max_tokens"]),
            )
        except Exception as exc:
            st.error(f"The request failed: {exc}")
            st.stop()

    st.session_state["intra_answer"] = kairo.get_response_text(response)

if st.session_state.get("intra_answer"):
    st.subheader("Answer")
    st.markdown(st.session_state["intra_answer"])
