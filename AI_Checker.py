# Author: Eshan Gurusinghe
# Email: engurusinghe91@gmail.com
# Version: 3.0 — Migrated from Ericsson ELI API to local Ollama stack -> Date: 2026-05-17
# Version: 3.1 — Dynamic model selector and model installation feature
#              — Install flow now runs check_disk_space() before showing install buttons
#              — BLOCKED state: install buttons hidden, error shown with exact shortfall
#              — WARNING state: warning shown, user must tick "I understand" checkbox to proceed
#              — OK state: space metrics shown, install proceeds normally
#              — Added "Uninstall a model" expander with at-least-one-model guard -> Date: 2026-05-18
# Changes from v2:
#   - llm_chat()
#   - Ollama connection status indicator
#   - Updated model dropdown           → Ollama model names
#   - Updated LLM response handling    → response is now a plain string, not a dict
#   - Model dropdown now populated dynamically via list_ollama_models()
#   - Added "Install a model" expander with Yes/No confirmation flow
#   - Added install_ollama_model() import
#   - Added KNOWN_OLLAMA_MODELS import for the install catalogue

import streamlit as st
from dotenv import load_dotenv
import numpy as np
import concurrent.futures
import os
import time

# Import backend functions
from backend.functions import (
    llm_chat,
    generate_output_file,
    check_for_uploaded_files,
    load_embeddings_from_file,
    get_most_relevant_docs,
    delete_remaining_resources,
    check_ollama_connection,
    list_ollama_models,
    install_ollama_model,
    uninstall_ollama_model,
    check_disk_space,
    KNOWN_OLLAMA_MODELS,
    MODEL_SIZES_GB,
)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

load_dotenv()
delete_remaining_resources()  # Clean up any leftover files from the previous session

# Session state defaults
if "chat_history"    not in st.session_state:
    st.session_state["chat_history"]    = []
if "texts"           not in st.session_state:
    st.session_state["texts"]           = []
if "vectors"         not in st.session_state:
    st.session_state["vectors"]         = np.array([])
if "upload_success"  not in st.session_state:
    st.session_state["upload_success"]  = False
if "installed_models" not in st.session_state:
    st.session_state["installed_models"] = []
if "confirm_install" not in st.session_state:
    st.session_state["confirm_install"] = False
if "model_to_install" not in st.session_state:
    st.session_state["model_to_install"] = ""
if "confirm_uninstall" not in st.session_state:
    st.session_state["confirm_uninstall"] = False
if "model_to_uninstall" not in st.session_state:
    st.session_state["model_to_uninstall"] = ""

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("AI Assistant — RCA")
st.write("Upload your reference documents, select a model, and ask questions.")

# ---------------------------------------------------------------------------
# Ollama connection status
# ---------------------------------------------------------------------------

ollama_ok, ollama_info = check_ollama_connection()

if ollama_ok:
    st.success(f"Ollama connected — {len(ollama_info)} model(s) available")
else:
    st.error(
        f"Cannot reach Ollama at {os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')}. "
        f"Make sure Ollama is running (`ollama serve`). Error: {ollama_info}"
    )

# ---------------------------------------------------------------------------
# Dynamic model selection
# ---------------------------------------------------------------------------

# Fetch installed models from Ollama on every page load.
# This reflects the real state — if the user just installed a model, it appears immediately.
installed_models = list_ollama_models()
st.session_state["installed_models"] = installed_models

if installed_models:
    model_choice = st.selectbox(
        "Select AI model",
        options=installed_models,
        help="Only models already installed in Ollama are listed here.",
    )
else:
    model_choice = None
    st.warning(
        "No models installed in Ollama yet. "
        "Use the **Install a model** section below to add one."
    )

# ---------------------------------------------------------------------------
# Model installation — expander with disk space check + Yes / No confirmation
# ---------------------------------------------------------------------------

with st.expander("Install a model"):
    st.write(
        "Select a model from the catalogue. "
        "The approximate download size is shown next to each name. "
        "Disk space on the Ollama models drive is checked automatically before downloading."
    )

    # Only models not yet installed appear in this list
    not_installed = {
        label: pull_name
        for label, pull_name in KNOWN_OLLAMA_MODELS.items()
        if pull_name not in installed_models
    }

    if not not_installed:
        st.success("All catalogue models are already installed.")
    else:
        selected_label = st.selectbox(
            "Choose a model to install",
            options=list(not_installed.keys()),
            key="install_model_select",
        )
        selected_pull_name = not_installed[selected_label]

        # ── Disk space check ──────────────────────────────────────────────
        space = check_disk_space(selected_pull_name)

        # Always show the three space metrics for full transparency
        m1, m2, m3 = st.columns(3)
        m1.metric("Required",         f"{space['required_gb']:.1f} GB")
        m2.metric("Free on drive",    f"{space['free_gb']:.1f} GB")
        m3.metric("Remaining after",  f"{space['remaining_pct']:.1f} %")

        # ── BLOCKED — not enough space ────────────────────────────────────
        if not space["can_install"]:
            st.error(
                f"**Cannot install — insufficient disk space.**\n\n"
                f"{space['message']}"
            )
            # Install buttons are intentionally NOT rendered here

        # ── WARNING — install fits but ≤ 10 % will remain ─────────────────
        elif space["warning"]:
            st.warning(
                f"**Low disk space warning.**\n\n"
                f"{space['message']}"
            )
            acknowledge = st.checkbox(
                "I understand the risk and want to proceed with the installation.",
                key="space_warning_ack",
            )
            if acknowledge:
                col_yes, col_no = st.columns([1, 5])
                with col_yes:
                    yes_button = st.button("Yes, install", key="install_yes")
                with col_no:
                    no_button  = st.button("No, cancel",   key="install_no")

                if yes_button:
                    st.session_state["confirm_install"]  = True
                    st.session_state["model_to_install"] = selected_pull_name
                if no_button:
                    st.session_state["confirm_install"]  = False
                    st.session_state["model_to_install"] = ""
                    st.info("Installation cancelled.")

        # ── OK — sufficient space ─────────────────────────────────────────
        else:
            st.success(space["message"])
            col_yes, col_no = st.columns([1, 5])
            with col_yes:
                yes_button = st.button("Yes, install", key="install_yes")
            with col_no:
                no_button  = st.button("No, cancel",   key="install_no")

            if yes_button:
                st.session_state["confirm_install"]  = True
                st.session_state["model_to_install"] = selected_pull_name
            if no_button:
                st.session_state["confirm_install"]  = False
                st.session_state["model_to_install"] = ""
                st.info("Installation cancelled.")

        # ── Execute download (shared by WARNING-acknowledged + OK paths) ──
        if (
            st.session_state.get("confirm_install")
            and st.session_state.get("model_to_install") == selected_pull_name
        ):
            with st.spinner(
                f"Downloading '{selected_pull_name}' — "
                f"this may take several minutes ({space['required_gb']:.1f} GB)..."
            ):
                success, message = install_ollama_model(selected_pull_name)

            st.session_state["confirm_install"]  = False
            st.session_state["model_to_install"] = ""

            if success:
                st.success(
                    f"**'{selected_pull_name}' installed successfully.** "
                )
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Installation failed: {message}")

# ---------------------------------------------------------------------------
# Model uninstall — at-least-one-model guard + Yes / No confirmation
# ---------------------------------------------------------------------------

with st.expander("Uninstall a model"):

    # ── Guard: block if no models installed ──────────────────────────────
    if len(installed_models) == 0:
        st.info("No models are installed yet.")

    # ── Guard: block if only one model remains ────────────────────────────
    elif len(installed_models) == 1:
        st.error(
            f"**Cannot uninstall '{installed_models[0]}'** — "
            "at least one model must remain installed for the app to function. "
            "Install a second model first, then you can remove this one."
        )

    # ── Safe to uninstall — two or more models present ────────────────────
    else:
        st.write(
            "Select the model you want to remove. "
            "The model files will be permanently deleted from your drive. "
            "You can re-install it at any time from the **Install a model** section above."
        )

        model_to_remove = st.selectbox(
            "Choose a model to uninstall",
            options=installed_models,
            key="uninstall_select",
        )

        # Show how much space will be recovered
        recover_gb = MODEL_SIZES_GB.get(model_to_remove)
        if recover_gb:
            st.info(
                f"Removing **'{model_to_remove}'** will free approximately "
                f"**{recover_gb:.1f} GB** of disk space."
            )
        else:
            st.info(
                f"**'{model_to_remove}'** is not in the built-in catalogue — "
                "disk space recovery amount is unknown."
            )

        st.warning(
            f"**This action is irreversible.** "
            f"'{model_to_remove}' will be permanently deleted and must be "
            "re-downloaded to use again. Are you sure?"
        )

        col_yes_u, col_no_u = st.columns([1, 5])
        with col_yes_u:
            yes_uninstall = st.button("Yes, uninstall", key="uninstall_yes")
        with col_no_u:
            no_uninstall  = st.button("No, cancel",     key="uninstall_no")

        if yes_uninstall:
            st.session_state["confirm_uninstall"]  = True
            st.session_state["model_to_uninstall"] = model_to_remove
        if no_uninstall:
            st.session_state["confirm_uninstall"]  = False
            st.session_state["model_to_uninstall"] = ""
            st.info("Uninstall cancelled.")

        # ── Execute removal ───────────────────────────────────────────────
        if (
            st.session_state.get("confirm_uninstall")
            and st.session_state.get("model_to_uninstall") == model_to_remove
        ):
            with st.spinner(f"Uninstalling '{model_to_remove}'..."):
                success, message = uninstall_ollama_model(model_to_remove)

            st.session_state["confirm_uninstall"]  = False
            st.session_state["model_to_uninstall"] = ""

            if success:
                st.success(                                  
                    f"**'{model_to_remove}' has been uninstalled.** "
                )
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Uninstall failed: {message}")

# ---------------------------------------------------------------------------
# Prompt input
# ---------------------------------------------------------------------------

user_input = st.text_area("Enter your prompt here:", key="user_input")

# ---------------------------------------------------------------------------
# Layout — three columns
# ---------------------------------------------------------------------------

col_left, col_middle, col_right = st.columns([1, 2, 1])

with col_middle:
    uploaded_files = st.file_uploader(
        "Upload reference files (TXT or CSV, max 200 MB each)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "default_uploader"),
    )

with col_left:
    chat_button = st.button("Ask AI", disabled=(not ollama_ok or model_choice is None))

with col_right:
    has_history  = len(st.session_state["chat_history"]) > 0
    print_button = st.button("Export Chat", disabled=not has_history)
    new_chat_button = st.button("New Chat")

# ---------------------------------------------------------------------------
# LLM call with timeout
# ---------------------------------------------------------------------------

def get_llm_response_with_timeout(func, timeout_seconds=240, **kwargs):
    """
    Run func(**kwargs) in a thread pool with a hard timeout.
    Returns the result string, or None on timeout.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return None

# ---------------------------------------------------------------------------
# Shared helpers — display chat and handle timeout
# ---------------------------------------------------------------------------

def display_chat_history():
    """Render the full chat history to the Streamlit page."""
    st.subheader("Chat History")
    for entry in st.session_state["chat_history"]:
        role = entry.get("role", "unknown")
        msg  = entry.get("content", "")
        if role == "User":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")


def handle_timeout():
    """Append a timeout notice and display the chat history."""
    st.error("4-minute LLM timeout — please rephrase your message and try again.")
    st.session_state["chat_history"].append({
        "role":    "Assistant",
        "content": "*Timeout: No response received within 4 minutes.*",
    })
    display_chat_history()


def handle_llm_response(response):
    """
    Process the LLM response string.
    Appends it to chat history and renders the updated history.
    Shows an error if the response indicates a failure.
    """
    if response is None:
        handle_timeout()
        return

    # llm_chat() returns a plain string — check for error prefix
    if isinstance(response, str) and response.startswith("An error occurred"):
        st.error(response)
        return

    st.success("Response generated.")
    st.session_state["chat_history"].append({
        "role":    "Assistant",
        "content": response,
    })
    display_chat_history()

# ---------------------------------------------------------------------------
# Core query processing
# ---------------------------------------------------------------------------

def process_user_input(
    query   = None,
    texts   = None,
    vectors = None,
):
    """
    Handle a single user query:
      1. Append the question to chat history.
      2. If uploaded documents exist, rerank them and inject the top results as context.
      3. Call the LLM and display the response.
    """
    query   = query   or user_input
    texts   = texts   or st.session_state["texts"]
    vectors = vectors if vectors is not None else st.session_state["vectors"]

    if not query.strip():
        st.warning("Please enter a prompt before clicking Ask AI.")
        return

    st.session_state["chat_history"].append({"role": "User", "content": query})

    # --- Path A: uploaded documents are available → RAG flow ---
    if texts:
        try:
            with st.spinner("Finding the most relevant sections from your documents..."):
                relevant_texts = get_most_relevant_docs(query=query, texts=texts)

            if relevant_texts:
                relevant_context = "\n\n".join(relevant_texts)
                print(f"Using context ({len(relevant_texts)} chunk(s)) for query.")
                prompt_messages = [
                    {
                        "role":    "system",
                        "content": (
                            "You are a knowledgeable assistant. Use the provided context "
                            "to answer the question as accurately as possible. "
                            "If the context does not contain enough information, say so clearly."
                        ),
                    },
                    {
                        "role":    "user",
                        "content": f"Context:\n{relevant_context}\n\nQuestion: {query}",
                    },
                ]
            else:
                # Reranker returned nothing useful — fall back to LLM-only
                st.warning("No strongly relevant sections found in your documents. Answering from model knowledge.")
                prompt_messages = [
                    {
                        "role":    "system",
                        "content": "You are a knowledgeable assistant.",
                    },
                    {
                        "role":    "user",
                        "content": f"Question: {query}",
                    },
                ]

            st.info("Generating response...")
            response = get_llm_response_with_timeout(
                llm_chat,
                timeout_seconds=240,
                messages=prompt_messages,
                model=model_choice,
                temperature=0.3,
            )
            handle_llm_response(response)

        except Exception as ex:
            st.error(f"An error occurred while processing your request: {ex}")

    # --- Path B: no uploaded documents → direct LLM query ---
    else:
        try:
            st.info("No documents uploaded — querying the model directly.")
            prompt_messages = [
                {
                    "role":    "system",
                    "content": "You are a knowledgeable assistant.",
                },
                {
                    "role":    "user",
                    "content": f"Question: {query}",
                },
            ]
            response = get_llm_response_with_timeout(
                llm_chat,
                timeout_seconds=240,
                messages=prompt_messages,
                model=model_choice,
                temperature=0.7,
            )
            handle_llm_response(response)

        except Exception as ex:
            st.error(f"An error occurred while querying the model: {ex}")

# ---------------------------------------------------------------------------
# File upload handling
# ---------------------------------------------------------------------------

if uploaded_files:
    try:
        with st.spinner("Processing uploaded files and creating embeddings..."):
            upload_result = check_for_uploaded_files(uploaded_files)

        if upload_result:
            texts, vectors = upload_result
            st.session_state["texts"]          = texts
            st.session_state["vectors"]        = vectors
            st.session_state["upload_success"] = True
        else:
            # Embeddings may already be cached — try loading from file
            cached = load_embeddings_from_file()
            if cached:
                st.session_state["texts"],   \
                st.session_state["vectors"] = cached
    except Exception as ex:
        st.error(f"An error occurred while uploading files: {ex}")
else:
    st.info("No files uploaded yet. You can still ask the model questions directly.")

if st.session_state.get("upload_success", False):
    st.success("Files uploaded and embeddings ready.")

# ---------------------------------------------------------------------------
# Chat export
# ---------------------------------------------------------------------------

def save_chat():
    """Write the current chat history to a .txt file and report the path."""
    try:
        output_path = generate_output_file(st.session_state["chat_history"])
        if output_path:
            st.success(f"Chat history saved to: {output_path}")
        else:
            st.error("Chat history is empty — nothing to export.")
    except Exception as ex:
        st.error(f"Failed to export chat history: {ex}")

# ---------------------------------------------------------------------------
# Button handlers
# ---------------------------------------------------------------------------

if chat_button:
    process_user_input()

if print_button:
    save_chat()

if new_chat_button:
    st.session_state.clear()
    st.session_state["uploader_key"]   = str(time.time())
    st.session_state["user_input"]     = ""
    st.info("Session cleared — ready for a new chat.")
    time.sleep(1)
    st.rerun()