# AI Assistant RCA

A locally hosted AI assistant for Root Cause Analysis (RCA). Upload reference documents, ask questions, and get context-aware answers — entirely on your own machine using [Ollama](https://ollama.ai). No data leaves your computer.

---

## Features

- Upload TXT or CSV files as a knowledge base for context-aware answers
- Semantic embedding and cross-encoder reranking for accurate document retrieval
- Dynamic model selector — lists only models currently installed in Ollama
- Install and uninstall Ollama models directly from the app UI
- Disk space validation before any model download — blocks if insufficient, warns if ≤ 10% drive space will remain
- Export chat history to a text file
- Fully offline — no external API calls

---

## Prerequisites

- Python 3.10 or higher — [python.org](https://www.python.org/downloads/)
- Ollama — [ollama.ai/download](https://ollama.ai/download)
- Git — [git-scm.com](https://git-scm.com)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/enguru91/AI_Assistant_RCA.git
cd AI_Assistant_RCA
```

### 2. Create and activate a virtual environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs these pinned packages:

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.57.0 | Web-based GUI |
| `python-dotenv` | 1.2.2 | Loads environment variables from `.env` |
| `numpy` | 2.4.5 | Numerical operations on embedding vectors |
| `scikit-learn` | 1.8.0 | Cosine similarity calculations |
| `sentence-transformers` | 5.5.0 | Local embedding and reranking models |
| `openai` | 2.37.0 | OpenAI-compatible client to communicate with Ollama |
| `torchvision` | 0.27.0 | open-source library in the PyTorch ecosystem specifically designed for computer vision tasks |

Verify installation:
```bash
# Windows
pip freeze | findstr /i "streamlit dotenv numpy scikit sentence openai"
# Mac / Linux
pip freeze | grep -iE "streamlit|dotenv|numpy|scikit|sentence|openai"
```

### 4. Pull at least one Ollama model

```bash
ollama pull llama3.1:8b    # recommended starting point (~4.7 GB)
```

Additional models can be installed later directly from the app UI.

Confirm Ollama is running and the model is available:
```bash
curl http://localhost:11434/api/tags
```

### 5. Create the `.env` file

Create a `.env` file in the project root:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

> `LLM_BASE_URL` points to Ollama's OpenAI-compatible endpoint. `LLM_API_KEY` is required by the OpenAI SDK but is ignored by Ollama — no real authentication occurs locally.

---

## Running the App

Make sure Ollama is running and your virtual environment is active, then:

```bash
streamlit run AI_Checker_v2.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Usage

| Step | Action |
|---|---|
| 1 | Select an installed model from the dropdown |
| 2 | *(Optional)* Upload TXT or CSV reference files |
| 3 | Type your question and click **Ask AI** |
| 4 | Click **Export Chat** to save the conversation as a `.txt` file |
| 5 | Click **New Chat** to clear history and start fresh |

**Install a model** — expand the section, pick a model from the built-in catalogue, review the disk space check, and confirm with Yes/No.

**Uninstall a model** — expand the section, select the model to remove, and confirm. At least one model must always remain installed.

---

## Project Structure

```
AI_Assistant_RCA/
├── AI_Checker_v2.py          # Streamlit app — UI and session management
├── backend/
│   └── functions_v2.py       # All backend logic
├── requirements.txt           # Pinned dependencies
├── .env                       # Local config — not committed to Git
├── .gitignore
├── README.md
└── docs/
    └── ARCHITECTURE.md
```

---

## .gitignore

```
.env
__pycache__/
venv/
*.pkl
NRJ-Dev/
```

---

## Author

**Eshan Gurusinghe** — engurusinghe91@gmail.com