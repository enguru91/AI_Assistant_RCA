# Architecture — AI Assistant RCA

## Overview

AI Assistant RCA is a locally hosted Retrieval-Augmented Generation (RAG) application. When the user uploads reference documents, the app converts them into searchable embedding vectors. On each query, the most relevant document sections are retrieved, reranked, and passed to a local LLM alongside the question. The LLM then generates a grounded answer.

Everything runs on the user's machine — no data is sent anywhere externally.

---

## System Diagram

```
┌─────────────────────────────────────────┐
│           Browser (User)                │
│        http://localhost:8501            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│        AI_Checker.py                 │
│        Streamlit UI Layer               │
│                                         │
│  • Ollama connection status             │
│  • Dynamic model dropdown               │
│  • Install / Uninstall model expanders  │
│  • File uploader                        │
│  • Prompt input & chat display          │
│  • Export Chat / New Chat buttons       │
└──────────────────┬──────────────────────┘
                   │ calls
                   ▼
┌─────────────────────────────────────────┐
│        backend/Functions.py          │
│        Backend Logic Layer              │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  Embedding   │  │   Reranking     │  │
│  │  Module      │  │   Module        │  │
│  │  bge-small   │  │   CrossEncoder  │  │
│  │  (local CPU) │  │   (local CPU)   │  │
│  └──────────────┘  └─────────────────┘  │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  LLM Chat    │  │  Model Manager  │  │
│  │  Module      │  │  list / install │  │
│  │  OpenAI SDK  │  │  uninstall      │  │
│  └──────┬───────┘  └─────────────────┘  │
└─────────┼───────────────────────────────┘
          │ HTTP /v1
          ▼
┌─────────────────────────────────────────┐
│        Ollama Server                    │
│        http://localhost:11434           │
│                                         │
│  llama3.1:8b  │  mistral  │  qwen2.5   │
└─────────────────────────────────────────┘
```

---

## Backend Functions — `Functions.py`

### Ollama Interface

| Function | Description |
|---|---|
| `check_ollama_connection()` | Calls `ollama_client.models.list()`. Returns `(True, model_list)` or `(False, error)`. Used to render the connection status banner in the UI. |
| `list_ollama_models()` | Returns a sorted list of installed model names. Called on every page load to populate the model dropdown dynamically. |
| `install_ollama_model(model_name)` | Runs `ollama pull <model>` via `subprocess`. Blocks until the download completes. 1-hour timeout. Returns `(bool, message)`. |
| `uninstall_ollama_model(model_name)` | Runs `ollama rm <model>` via `subprocess`. 120-second timeout. Returns `(bool, message)`. |
| `check_disk_space(model_name)` | Reads drive stats with `shutil.disk_usage()` on the OLLAMA_MODELS path. Returns a dict with three possible outcomes — see below. |

**`check_disk_space()` outcomes:**

| State | Condition | Effect in UI |
|---|---|---|
| BLOCKED | `free_gb < required_gb` | Error shown, install buttons not rendered |
| WARNING | Install fits but ≤ 10% of drive remains | Warning shown, user must tick a checkbox to proceed |
| OK | Sufficient space | Success message, install buttons shown normally |

### Embedding Module

| Function | Description |
|---|---|
| `create_embeddings_local(texts)` | Encodes a list of strings into 384-dimensional vectors using `BAAI/bge-small-en-v1.5`. Runs on CPU. Returns `{"vectors": [...]}`. |
| `create_embeddings()` | Deletes old `.pkl` files, loads resource texts, calls `create_embeddings_local()`, saves result to a new timestamped `.pkl`. Returns `(texts, np.array)`. |
| `load_embeddings_from_file()` | Loads the latest `.pkl` from the embeddings directory. Falls back to `create_embeddings()` if none exists. Cached by `@st.cache_data`. |

### Reranking Module

| Function | Description |
|---|---|
| `local_score_rank(query, texts)` | Scores every `(query, document)` pair using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Returns `{"scores": [...]}`. |
| `get_most_relevant_docs(query, texts)` | Calls `local_score_rank()` and returns documents sorted by score, highest first. |

### File Management

| Function | Description |
|---|---|
| `delete_remaining_resources()` | Deletes leftover `ResourcesFile_`, `current_checksum_`, and `current_embeddings_` files from the previous session. Called once at startup. |
| `save_uploaded_files(uploaded_files)` | Saves uploaded Streamlit file objects to disk and computes an MD5 checksum of the saved content. |
| `compute_resources_checksum(directory)` | MD5 hash across all `ResourcesFile_*.txt` files. Used to detect whether uploaded files have changed. |
| `check_for_uploaded_files(uploaded_files)` | Orchestrates the upload flow — compares checksums and either loads cached embeddings or creates new ones. |
| `generate_output_file(chat_history)` | Writes the full chat history to a timestamped `.txt` file in the Outputs/Chats directory. |

---

## Data Flows

### Query with uploaded documents (RAG)

```
User submits a question
        │
        ▼
get_most_relevant_docs()
  — CrossEncoder scores each (query, document) pair
  — Returns documents sorted by relevance
        │
        ▼
llm_chat()
  — Builds prompt: system instruction + context + question
  — Sends to Ollama via OpenAI SDK → /v1/chat/completions
        │
        ▼
Response string appended to chat_history
UI re-renders with updated conversation
```

### Query without documents (direct LLM)

```
User submits a question
        │
        ▼
llm_chat()
  — Sends question directly with no context
        │
        ▼
Response string appended to chat_history
```

### File upload

```
User uploads TXT / CSV files
        │
        ▼
check_for_uploaded_files()
  — Saves files to ProcedureFiles/
  — Computes MD5 checksum
        │
        ├── Checksum unchanged → load cached .pkl
        │
        └── Checksum changed  → create_embeddings()
                                  encode all texts → vectors
                                  save new .pkl
        │
        ▼
(texts, vectors) stored in st.session_state
Ready for queries
```

### Model install

```
User selects a model from the catalogue
        │
        ▼
check_disk_space()
  — Resolves OLLAMA_MODELS path
  — Reads drive stats via shutil.disk_usage()
  — Compares free space against MODEL_SIZES_GB
        │
        ├── BLOCKED  → error, no buttons rendered
        ├── WARNING  → warning + checkbox required before buttons appear
        └── OK       → success message, buttons rendered
        │
User clicks "Yes, install"
        │
        ▼
install_ollama_model()   — runs `ollama pull <model>`
        │
        ▼
Success / failure message shown
User refreshes page → model appears in dropdown
```

---

## File Storage Layout

```
NRJ-Dev/Include/
├── ProcedureFiles/
│   ├── ResourcesFile_<timestamp>_1.txt     ← uploaded files saved here
│   └── current_checksum_<timestamp>.txt    ← MD5 of current resource files
└── Outputs/
    ├── Embeddings/
    │   └── current_embeddings_<timestamp>.pkl   ← cached vectors + texts
    └── Chats/
        └── chat_output_<timestamp>.txt           ← exported conversations
```

All files under `NRJ-Dev/` are excluded from version control via `.gitignore`.

---

## Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit 1.57.0 |
| LLM runtime | Ollama (local) |
| LLM API client | `openai` 2.37.0 → Ollama `/v1` endpoint |
| Embedding model | `sentence-transformers` — BAAI/bge-small-en-v1.5 (~130 MB, CPU) |
| Reranking model | `sentence-transformers` — cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB, CPU) |
| Disk space check | Python `shutil.disk_usage()` |
| Model management | Python `subprocess` → `ollama pull` / `ollama rm` |
| Embedding cache | Python `pickle` |
| Config | `python-dotenv` |