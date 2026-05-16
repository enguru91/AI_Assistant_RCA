# AI Assistant RCA

A locally hosted AI assistant for Root Cause Analysis (RCA). Upload your reference documents, ask questions, and get context-aware answers — all running privately on your own machine using [Ollama](https://ollama.ai).

---

## Features

- Upload TXT or CSV reference documents as the knowledge base
- Automatic text embedding and semantic reranking for relevant context retrieval
- Multi-turn chat interface built with Streamlit
- Supports multiple local LLM models via Ollama
- Export chat history to a text file
- Fully offline — no data leaves your machine

---

## Prerequisites

Make sure the following are installed before proceeding:

- Python 3.10 or higher — [python.org](https://www.python.org/downloads/)
- Ollama — [ollama.ai](https://ollama.ai/download)
- Git — [git-scm.com](https://git-scm.com)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/enguru91/AI_Assistant_RCA.git
cd AI_Assistant_RCA
```

### 2. Create and Activate a Virtual Environment

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

### 3. Install Dependencies

The `requirements.txt` file contains all required packages with their pinned versions to ensure a stable, reproducible environment.

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.57.0 | Web-based GUI |
| `python-dotenv` | 1.2.2 | Load environment variables from `.env` |
| `numpy` | 2.4.5 | Numerical operations on embeddings |
| `scikit-learn` | 1.8.0 | Cosine similarity calculations |
| `sentence-transformers` | 5.5.0 | Local text embedding and reranking models |
| `openai` | 2.37.0 | OpenAI-compatible client to communicate with Ollama |

To verify all packages installed correctly:
```bash
pip freeze | findstr /i "streamlit dotenv numpy scikit sentence openai"   # Windows
pip freeze | grep -i "streamlit\|dotenv\|numpy\|scikit\|sentence\|openai"  # Mac/Linux
```

### 4. Pull Ollama Models

Start Ollama, then pull the models you want to use. Each model is several GB, so choose based on your available disk space:

```bash
ollama pull llama3.1:8b       # recommended starting point (~4.7 GB)
ollama pull mistral:latest    # (~4.1 GB)
ollama pull qwen2.5:7b        # (~4.4 GB)
ollama pull phi4:latest       # (~9.1 GB)
```

Verify Ollama is running and models are available:
```bash
curl http://localhost:11434/api/tags
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
```

> ⚠️ The `.env` file is listed in `.gitignore` and will never be committed to GitHub.

---

## Running the App

Make sure your virtual environment is activated and Ollama is running, then:

```bash
streamlit run AI_Checker_v2.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## Usage

1. **Select a model** from the dropdown at the top of the page
2. **Upload reference files** (TXT or CSV, max 200 MB each) using the file uploader
   - The app creates semantic embeddings from your files automatically
   - If the same files are uploaded again, cached embeddings are reused
3. **Type your question** in the prompt box
4. **Click "Ask AI"** to get a context-aware response
5. **Export Chat** to save the conversation as a text file
6. **New Chat** to clear history and start fresh

---

## Project Structure

```
AI_Assistant_RCA/
├── AI_Checker_v2.py          # Main Streamlit application
├── backend/
│   └── functions_v2.py       # All backend logic (embeddings, LLM, reranking)
├── requirements.txt           # Pinned Python dependencies
├── .env                       # Local environment config (not committed)
├── .gitignore
├── README.md
├── docs/
│   └── ARCHITECTURE.md       # System architecture documentation
└── Includes/
        ├── ProcedureFiles/   # Uploaded resource files (auto-managed)
        └── Outputs/
            ├── Embeddings/   # Cached embedding pickle files
            └── Chats/        # Exported chat history files
```

---

## .gitignore

Ensure these are excluded from version control:

```
.env
__pycache__/
venv/
*.pkl
Includes/
```

---

## Author

**Eshan Gurusinghe**
engurusinghe91@gmail.com
