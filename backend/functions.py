# Author: Eshan Gurusinghe
# Email: engurusinghe91@gmail.com
# Version: 3.0 — Migrated from Ericsson ELI API to local Ollama stack
# Date: 2026-05-17
#
# Changes from v2:
#   - Removed all Ericsson ELI API dependencies (eli package, ELIClient, eli-key)
#   - create_embeddings_local()  using sentence-transformers
#   - local_score_rank()         using CrossEncoder
#   - llm_chat()                 using Ollama via OpenAI SDK
#   - Added check_ollama_connection() for GUI health status

import os
import pickle
import glob
import hashlib
import numpy as np
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Environment & path constants
# ---------------------------------------------------------------------------

load_dotenv()

resources_path   = "./Include/ProcedureFiles/"
embeddings_path  = "./Include/Outputs/Embeddings/"
output_dir       = "./Include/Outputs/Chats/"
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix  = "ResourcesFile_"  + current_datetime
checksum_file    = "current_checksum_" + current_datetime

# Ensure output directories exist on startup
os.makedirs(resources_path,  exist_ok=True)
os.makedirs(embeddings_path, exist_ok=True)
os.makedirs(output_dir,      exist_ok=True)

# ---------------------------------------------------------------------------
# Local model initialisation
# ---------------------------------------------------------------------------

# Embedding model — converts text chunks into 384-dimensional vectors.
# Auto-downloads (~130 MB) on first use, then cached locally.
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Cross-encoder reranking model — scores (query, document) pairs for relevance.
# Auto-downloads (~80 MB) on first use, then cached locally.
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Ollama client initialisation (OpenAI-compatible)
# ---------------------------------------------------------------------------

# Ollama exposes an OpenAI-compatible API at /v1.
# LLM_API_KEY is required by the SDK but ignored by Ollama — no real auth.
ollama_client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY",   "ollama"),
)

# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama_connection():
    """
    Check whether the Ollama server is reachable and return available models.
    Returns (True, [model_names]) on success, or (False, error_message) on failure.
    """
    try:
        models = ollama_client.models.list()
        model_names = [m.id for m in models.data]
        print(f"Ollama connected. Available models: {model_names}")
        return True, model_names
    except Exception as ex:
        print(f"Ollama connection failed: {ex}")
        return False, str(ex)

# ---------------------------------------------------------------------------
# File & resource management
# ---------------------------------------------------------------------------

def delete_remaining_resources():
    """
    Delete leftover ResourcesFile_, current_checksum_, and current_embeddings_
    files from previous sessions. Called once at app startup.
    """
    cleanup_targets = [
        (resources_path,  "ResourcesFile_",      ".txt"),
        (resources_path,  "current_checksum_",   ".txt"),
        (embeddings_path, "current_embeddings_", ".pkl"),
    ]
    for directory, prefix, extension in cleanup_targets:
        if not os.path.isdir(directory):
            print(f"Directory does not exist, skipping cleanup: {directory}")
            continue
        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith(extension):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as ex:
                    print(f"Failed to delete {file_path}: {ex}")


def save_uploaded_files(uploaded_files):
    """
    Save uploaded Streamlit file objects to resources_path, compute an MD5
    checksum of the saved content, and write it to the checksum file.
    Returns the checksum string on success, or None on failure.
    """
    # Remove any existing ResourcesFile_ entries before saving new ones
    pattern = os.path.join(resources_path, "ResourcesFile_*")
    for file in glob.glob(pattern):
        try:
            os.remove(file)
            print(f"Deleted old resource file: {file}")
        except Exception as ex:
            print(f"Error deleting resource file {file}: {ex}")

    # Write each uploaded file to disk with a timestamped name
    for index, uploaded_file in enumerate(uploaded_files):
        new_filename = f"{filename_prefix}_{index + 1}.txt"
        file_path = os.path.join(resources_path, new_filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        print(f"Saved uploaded file: {file_path}")

    # Compute checksum of the newly saved files
    new_checksum = compute_resources_checksum(resources_path)
    if not new_checksum:
        print("Error: checksum computation returned empty.")
        return None

    # Write checksum into any existing current_checksum_ file in resources_path
    for filename in sorted(os.listdir(resources_path)):
        if filename.startswith("current_checksum_") and filename.endswith(".txt"):
            filepath = os.path.join(resources_path, filename)
            try:
                with open(filepath, "w") as f:
                    f.write(new_checksum)
                print(f"Checksum written to {filepath}: {new_checksum}")
            except Exception as ex:
                print(f"Error writing checksum to {filepath}: {ex}")

    return new_checksum


def compute_resources_checksum(directory):
    """
    Compute an MD5 hash across all ResourcesFile_ .txt files in the directory.
    Returns the hex digest string, or None if no files are found.
    """
    hash_md5 = hashlib.md5()
    found_any = False
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("ResourcesFile_") and filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            found_any = True
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    if not found_any:
        print("No ResourcesFile_ files found for checksum computation.")
        return None
    return hash_md5.hexdigest()


def check_exists_checksum_file(directory):
    """
    Find the current checksum file in the given directory.
    If none exists, create a new empty one.
    Returns the full path to the checksum file.
    """
    try:
        matching_files = [
            f for f in os.listdir(directory)
            if f.startswith("current_checksum_") and f.endswith(".txt")
        ]
        if matching_files:
            matching_files.sort()
            return os.path.join(directory, matching_files[0])

        # No checksum file found — create a new empty one
        print("No checksum file found. Creating a new one.")
        new_checksum_filename = f"current_checksum_{current_datetime}.txt"
        new_checksum_filepath = os.path.join(directory, new_checksum_filename)
        with open(new_checksum_filepath, "w") as f:
            f.write("")
        print(f"Created new empty checksum file: {new_checksum_filepath}")
        return new_checksum_filepath

    except FileNotFoundError:
        print(f"Directory does not exist: {directory}")
        return None
    except Exception as ex:
        print(f"Error searching for checksum file: {ex}")
        return None


def check_for_uploaded_files(uploaded_files):
    """
    Orchestrate the upload flow:
      1. Save uploaded files to disk
      2. Compare new checksum against the stored one
      3. Return cached embeddings if unchanged, or create new ones if changed
    """
    current_checksum_path = check_exists_checksum_file(resources_path)

    if not uploaded_files:
        print("No uploaded files detected.")
        return None

    print("Uploaded files detected.")

    # Read the stored checksum (if any)
    current_checksum_value = None
    if current_checksum_path and os.path.exists(current_checksum_path):
        try:
            with open(current_checksum_path, "r") as f:
                current_checksum_value = f.read().strip()
            print(f"Stored checksum: {current_checksum_value}")
        except Exception as ex:
            print(f"Error reading checksum file: {ex}")

    # Save files and compute new checksum
    new_checksum_value = save_uploaded_files(uploaded_files)
    print(f"New checksum: {new_checksum_value}")

    if current_checksum_value and current_checksum_value == new_checksum_value:
        print("Checksum matches — loading cached embeddings.")
        return load_embeddings_from_file()

    print("Checksum differs — creating new embeddings.")
    embeddings_data = create_embeddings()

    # Persist the new checksum
    if current_checksum_path:
        try:
            with open(current_checksum_path, "w") as f:
                f.write(new_checksum_value)
            print(f"Updated stored checksum to: {new_checksum_value}")
        except Exception as ex:
            print(f"Error updating checksum file: {ex}")

    return embeddings_data


def load_resources(directory):
    """
    Load all ResourcesFile_ .txt files from the given directory.
    Returns a list of strings (one per file).
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.startswith("ResourcesFile_") and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    print(f"Loaded {len(texts)} resource file(s).")
    return texts

# ---------------------------------------------------------------------------
# Embedding — local sentence-transformers model
# ---------------------------------------------------------------------------

def create_embeddings_local(texts):
    """
    Encode a list of text strings into embedding vectors using the local
    BAAI/bge-small-en-v1.5 model. Runs entirely on CPU with no API call.
    Returns a dict {"vectors": list_of_lists} to match the original interface.
    """
    try:
        vectors = embed_model.encode(texts, convert_to_numpy=True)
        print(f"Created embeddings for {len(texts)} text(s). Shape: {vectors.shape}")
        return {"vectors": vectors.tolist()}
    except Exception as ex:
        print(f"Error during embedding: {ex}")
        return {}


def create_embeddings(embeddings_dir=embeddings_path):
    """
    Delete any existing embedding .pkl files, create fresh embeddings from
    all resource files, and save them to a new timestamped .pkl file.
    Returns (texts, numpy_vectors) tuple.
    """
    # Remove old embedding files
    for file in glob.glob(os.path.join(embeddings_dir, "current_embeddings_*")):
        try:
            os.remove(file)
            print(f"Deleted old embedding file: {file}")
        except Exception as ex:
            print(f"Error deleting embedding file {file}: {ex}")

    print("Creating new embeddings...")
    texts = load_resources(resources_path)
    if not texts:
        print("No resource texts found. Cannot create embeddings.")
        return [], np.array([])

    embeddings = create_embeddings_local(texts)

    if not embeddings.get("vectors"):
        print("Embedding creation failed.")
        return [], np.array([])

    # Save to a new timestamped .pkl file
    new_filename  = f"current_embeddings_{current_datetime}.pkl"
    new_file_path = os.path.join(embeddings_dir, new_filename)
    with open(new_file_path, "wb") as f:
        pickle.dump({"texts": texts, "vectors": embeddings["vectors"]}, f)
    print(f"Saved new embeddings to: {new_filename}")
    return texts, np.array(embeddings["vectors"])


@st.cache_data
def load_embeddings_from_file(embeddings_dir=embeddings_path):
    """
    Load embeddings from the most recent .pkl file in embeddings_dir.
    Falls back to creating new embeddings if no .pkl file is found.
    Result is cached by Streamlit to avoid reloading on every rerun.
    """
    pkl_files = [
        f for f in os.listdir(embeddings_dir)
        if f.startswith("current_embeddings_") and f.endswith(".pkl")
    ]
    if pkl_files:
        pkl_files.sort()
        embeddings_file = os.path.join(embeddings_dir, pkl_files[-1])
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cached embeddings from: {embeddings_file}")
        return data["texts"], np.array(data["vectors"])

    print("No cached embeddings found. Creating new ones.")
    return create_embeddings(embeddings_dir)

# ---------------------------------------------------------------------------
# Reranking — local CrossEncoder model
# ---------------------------------------------------------------------------

def local_score_rank(query, texts):
    """
    Score each (query, document) pair using the local CrossEncoder model.
    Returns a dict {"scores": list_of_floats} to match the original interface.
    Runs entirely on CPU with no API call.
    """
    try:
        pairs  = [(query, doc) for doc in texts]
        scores = rerank_model.predict(pairs)
        print(f"Reranking complete. Scores: {scores}")
        return {"scores": scores.tolist()}
    except Exception as ex:
        print(f"Error during reranking: {ex}")
        return {}


def get_most_relevant_docs(query, texts):
    """
    Return the uploaded documents sorted by relevance to the query,
    most relevant first, using the local CrossEncoder reranker.
    """
    if not texts:
        print("No texts available for reranking.")
        return []

    reranked_response = local_score_rank(query=query, texts=texts)
    if not reranked_response:
        return []

    scores = reranked_response.get("scores", [])
    reranked_documents = [
        texts[idx]
        for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ]
    print(f"Reranked {len(reranked_documents)} document(s).")
    return reranked_documents

# ---------------------------------------------------------------------------
# LLM chat — Ollama via OpenAI-compatible client
# ---------------------------------------------------------------------------

def llm_chat(messages=[], model="llama3.1:8b", temperature=0.3, max_tokens=4096):
    """
    Send a list of messages to the Ollama LLM and return the response as a string.

    Parameters
    ----------
    messages    : list of {"role": str, "content": str} dicts
    model       : Ollama model name (must be pulled via `ollama pull <model>`)
    temperature : creativity control — lower = more deterministic
    max_tokens  : maximum tokens in the response

    Returns
    -------
    str — the assistant's reply, or an error message string on failure
    """
    # Larger models benefit from more tokens
    if model in ("mistral:latest", "phi4:latest"):
        max_tokens = 8192

    try:
        response = ollama_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        print(f"LLM response received from model: {model}")
        return content
    except Exception as ex:
        print(f"Error during LLM chat: {ex}")
        return f"An error occurred while contacting Ollama: {ex}"

# ---------------------------------------------------------------------------
# Chat export
# ---------------------------------------------------------------------------

def generate_output_file(chat_history):
    """
    Write the full chat history to a timestamped .txt file in output_dir.
    Returns the file path on success, or None if chat_history is empty.
    """
    if not chat_history:
        print("Chat history is empty — nothing to export.")
        return None

    filename = f"chat_output_{current_datetime}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for entry in chat_history:
            role    = entry.get("role",    "unknown").capitalize()
            content = entry.get("content", "").strip()
            f.write(f"{role}: {content}\n\n")

    print(f"Chat history exported to: {filepath}")
    return filepath