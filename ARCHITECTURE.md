# Architecture — AI Assistant RCA

## Overview

AI Assistant RCA is a locally hosted Retrieval-Augmented Generation (RAG) application. The user uploads reference documents, which are converted into searchable embeddings. When the user asks a question, the most relevant document sections are retrieved and passed alongside the question to a local LLM, which generates the final answer.

Everything runs on the user's machine. No data is sent to external services.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User (Browser)                     │
│              http://localhost:8501                  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit GUI Layer                    │
│              AI_Checker_v2.py                       │
│                                                     │
│  - Model selector dropdown                          │
│  - File uploader                                    │
│  - Prompt input                                     │
│  - Chat history display                             │
│  - Export / New Chat buttons                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Backend Logic Layer                    │
│              backend/functions_v2.py                │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Embedding  │  │  Reranking   │  │  LLM Chat │  │
│  │  Module     │  │  Module      │  │  Module   │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │
└─────────┼────────────────┼────────────────┼─────────┘
          │                │                │
          ▼                ▼                ▼
┌──────────────────┐  ┌──────────┐  ┌─────────────────┐
│ sentence-        │  │ sentence-│  │ Ollama Server   │
│ transformers     │  │ transform│  │ localhost:11434  │
│ BAAI/bge-small   │  │ CrossEnc │  │                 │
│ (local, CPU)     │  │ (local)  │  │ llama3.1:8b     │
└──────────────────┘  └──────────┘  │ mistral:latest  │
                                     │ qwen2.5:7b      │
                                     │ phi4:latest     │
                                     └─────────────────┘
```

---

## Component Breakdown

### 1. Streamlit GUI — `AI_Checker_v2.py`

The entry point and user interface. Responsibilities:

- Renders all UI elements (dropdowns, inputs, buttons, chat display)
- Manages Streamlit session state (`chat_history`, `texts`, `vectors`)
- Calls backend functions and handles their responses
- Manages the LLM response timeout using `concurrent.futures.ThreadPoolExecutor`

Session state variables:

| Variable | Type | Purpose |
|---|---|---|
| `chat_history` | list of dicts | Stores all user/assistant messages |
| `texts` | list of str | Raw text content from uploaded documents |
| `vectors` | numpy array | Embedding vectors corresponding to `texts` |
| `upload_success` | bool | Tracks whether file upload completed |

---

### 2. Backend Logic — `backend/functions_v2.py`

All non-UI logic lives here. It is split into four functional areas:

#### A. File & Checksum Management
- `save_uploaded_files()` — saves uploaded files to `ProcedureFiles/` with a timestamped name
- `compute_resources_checksum()` — computes an MD5 hash of all resource files to detect changes
- `check_exists_checksum_file()` — finds or creates the current checksum file
- `check_for_uploaded_files()` — orchestrates the upload flow: compares checksums and decides whether to reuse cached embeddings or create new ones
- `delete_remaining_resources()` — cleans up leftover files from previous sessions on startup

#### B. Embedding Module
- `create_embeddings_local()` — encodes all resource texts into vectors using `BAAI/bge-small-en-v1.5` via `sentence-transformers`
- `create_embeddings()` — manages deletion of old embedding files, calls the encoder, saves results to a `.pkl` file
- `load_embeddings_from_file()` — loads the latest cached `.pkl` file; falls back to creating new embeddings if none exist

#### C. Reranking Module
- `local_score_rank()` — uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score each (query, document) pair
- `get_most_relevant_docs()` — calls `local_score_rank()` and returns documents sorted by relevance score

#### D. LLM Chat Module
- `llm_chat()` — sends the structured prompt messages to Ollama via the OpenAI-compatible `/v1/chat/completions` endpoint and returns the response text
- `generate_output_file()` — writes the full chat history to a timestamped `.txt` file

---

### 3. Embedding Model — `BAAI/bge-small-en-v1.5`

- Runs entirely locally on CPU via `sentence-transformers`
- Converts text chunks into 384-dimensional vectors
- Used to represent both the uploaded documents and (optionally) the user query in the same vector space
- Model files are cached locally after first download (~130 MB)

---

### 4. Reranking Model — `cross-encoder/ms-marco-MiniLM-L-6-v2`

- Runs entirely locally on CPU via `sentence-transformers`
- Takes `(query, document)` pairs and outputs a relevance score for each
- More accurate than cosine similarity alone — it reads both texts together rather than comparing independent vectors
- Model files are cached locally after first download (~80 MB)

---

### 5. Ollama LLM Server

- Runs as a background service on `http://localhost:11434`
- Exposes an OpenAI-compatible API at `/v1` — this is what the `openai` Python SDK connects to
- The `/v1` path is not a separate server; it is the same Ollama process responding in OpenAI's JSON format
- The `api_key` value `"ollama"` is required by the SDK but is ignored by Ollama — no real authentication occurs locally

---

## Data Flow — Single Query

```
User types a question
        │
        ▼
[ AI_Checker_v2.py ]
  Appends question to chat_history
        │
        ▼
[ get_most_relevant_docs() ]
  Passes (query, each document) pairs to CrossEncoder
        │
        ▼
[ local_score_rank() ]
  Returns relevance scores → documents sorted by score
        │
        ▼
[ llm_chat() ]
  Builds prompt:
    system: "You are a knowledgeable assistant."
    user:   "Context: <top ranked docs>\nQuestion: <query>"
  Sends to Ollama via OpenAI SDK
        │
        ▼
[ Ollama ]
  Selected model generates a response
        │
        ▼
[ AI_Checker_v2.py ]
  Appends response to chat_history
  Renders updated chat in the browser
```

---

## Data Flow — File Upload

```
User uploads TXT/CSV files
        │
        ▼
[ check_for_uploaded_files() ]
  Saves files to ProcedureFiles/
  Computes MD5 checksum of new files
        │
        ├── Checksum matches existing? ──► Load cached .pkl embeddings
        │
        └── Checksum differs? ──────────► Delete old .pkl
                                          Call create_embeddings()
                                          Encode all texts → vectors
                                          Save new .pkl to Embeddings/
                                          Return (texts, vectors)
        │
        ▼
[ AI_Checker_v2.py ]
  Stores texts and vectors in session state
  Ready for queries
```

---

## File Storage Layout

```
Includes/
├── ProcedureFiles/
│   ├── ResourcesFile_<timestamp>_1.txt    ← saved uploaded files
│   ├── ResourcesFile_<timestamp>_2.txt
│   └── current_checksum_<timestamp>.txt   ← MD5 of current resource files
└── Outputs/
    ├── Embeddings/
    │   └── current_embeddings_<timestamp>.pkl  ← cached vectors + texts
    └── Chats/
        └── chat_output_<timestamp>.txt          ← exported chat history
```

All files under `Includes/` are excluded from version control via `.gitignore`.

---

## Technology Stack

| Layer | Technology |
|---|---|
| GUI | Streamlit 1.57.0 |
| Embedding model | `sentence-transformers` — BAAI/bge-small-en-v1.5 |
| Reranking model | `sentence-transformers` — cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM runtime | Ollama (local) |
| LLM API client | `openai` SDK 2.37.0 pointed at Ollama's `/v1` endpoint |
| Similarity | `scikit-learn` cosine similarity |
| Numerical ops | `numpy` |
| Config | `python-dotenv` |
| Serialisation | `pickle` (embedding cache) |
