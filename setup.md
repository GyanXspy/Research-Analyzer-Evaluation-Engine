# 🛠️ Installation & Setup Guide

This guide walks you through the step-by-step process of installing, configuring, running, and testing the **AI Research Analyzer & Evaluation Engine**.

---

## 📋 System Prerequisites

Ensure you have the following installed on your system before proceeding:

| Requirement | Recommended Version | Notes |
|---|---|---|
| **Python** | `3.11` or `3.12` | Built & verified with Python 3.12 64-bit |
| **Operating System** | Windows 10/11, macOS, Linux | Cross-platform compatible |
| **Package Manager** | `pip` (latest version) | Run `python -m pip install --upgrade pip` |
| **Git** | `2.30+` | Required for cloning the repository |
| **API Keys** | Google Gemini **or** OpenAI | At least one valid API key is required |

---

## 📥 1. Clone the Repository

Clone the project to your local workstation:

```bash
git clone https://github.com/your-username/ai-evaluation-engine.git
cd "ai-evaluation-engine"
```

---

## 🐍 2. Create and Activate a Virtual Environment

It is recommended to use an isolated Python virtual environment to avoid dependency conflicts.

### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> *Tip: If you encounter an execution policy error on PowerShell, run:*
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 3. Install Dependencies

Install all project dependencies specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Key Libraries Installed:
- **FastAPI & Uvicorn**: High-concurrency async REST API framework.
- **Streamlit**: Interactive frontend UI.
- **LangChain & LangGraph**: Stateful graph-based RAG workflow orchestration.
- **LangChain Google GenAI / OpenAI**: LLM and embedding integrations.
- **ChromaDB**: High-performance persistent vector database.
- **PyMuPDF (`fitz`)**: Fast, high-fidelity PDF text and page metadata extraction.
- **Pydantic v2**: Type validation, configuration management, and schema enforcement.
- **Pytest & HTTPX**: Testing suite and async test client.

---

## ⚙️ 4. Environment Configuration

The engine uses a `.env` file at the root of the project to manage settings, API keys, and models.

1. Copy the sample environment file:
   ```bash
   cp .env.example .env      # macOS / Linux
   copy .env.example .env    # Windows CMD
   Copy-Item .env.example .env # Windows PowerShell
   ```

2. Open `.env` in your editor and configure your provider:

### Option A: Google Gemini (Default / Recommended)
Get an API key from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

```env
# ==========================================
# LLM Provider Configuration
# ==========================================
LLM_PROVIDER=google
GOOGLE_API_KEY=your_gemini_api_key_here

# Core Models
LLM_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
EVAL_LLM_MODEL=gemini-3.5-flash-lite

# Model Parameters
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048

# Vector Store & Chunking
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=4

# Evaluation Settings
EVALUATION_PASS_THRESHOLD=0.7
```

### Option B: OpenAI
Get an API key from OpenAI Platform: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

```env
# ==========================================
# LLM Provider Configuration
# ==========================================
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# Core Models
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EVAL_LLM_MODEL=gpt-4o-mini

# Model Parameters
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048

# Vector Store & Chunking
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=4

# Evaluation Settings
EVALUATION_PASS_THRESHOLD=0.7
```

---

## 🚀 5. Running the Application

The system consists of two processes:
1. **Backend API (FastAPI)** on port `8000`
2. **Frontend UI (Streamlit)** on port `8501`

### Step 5.1: Start the Backend Server
In your first activated terminal, run:

```bash
python main.py
```
*Alternatively, using uvicorn directly:*
```bash
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

Once running, you will see:
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

- **Health Check**: Visit `http://localhost:8000/api/health`
- **Interactive Swagger Docs**: Visit `http://localhost:8000/docs`
- **ReDoc Alternative Docs**: Visit `http://localhost:8000/redoc`

### Step 5.2: Start the Streamlit Frontend
Open a **second terminal**, activate your virtual environment, and run:

```bash
streamlit run streamlit_app.py
```

The Streamlit dashboard will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 6. Running Automated Tests

A comprehensive suite of unit and integration tests is included. To execute all tests:

```bash
pytest -v tests/
```

### Expected Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.x, pytest-8.x.x
collected 16 items

tests/test_config.py::test_settings_load PASSED                          [  6%]
tests/test_config.py::test_settings_provider_switch PASSED               [ 12%]
tests/test_schemas.py::test_chat_request_validation PASSED               [ 18%]
tests/test_schemas.py::test_evaluation_request_validation PASSED         [ 25%]
tests/test_schemas.py::test_evaluation_result_validation PASSED          [ 31%]
tests/test_schemas.py::test_document_metadata_coercion PASSED            [ 37%]
tests/test_evaluators.py::test_faithfulness_evaluator PASSED             [ 43%]
tests/test_evaluators.py::test_answer_relevance_evaluator PASSED         [ 50%]
tests/test_evaluators.py::test_context_relevance_evaluator PASSED        [ 56%]
tests/test_evaluators.py::test_correctness_evaluator PASSED              [ 62%]
tests/test_evaluators.py::test_completeness_evaluator PASSED             [ 68%]
tests/test_evaluators.py::test_evaluation_engine_aggregation PASSED       [ 75%]
tests/test_api.py::test_health_endpoint PASSED                           [ 81%]
tests/test_api.py::test_evaluate_endpoint PASSED                         [ 87%]
tests/test_api.py::test_documents_endpoint PASSED                        [ 93%]
tests/test_api.py::test_invalid_evaluate_payload PASSED                   [100%]

============================= 16 passed in 2.15s ==============================
```

---

## 🔍 7. Verification Walkthrough

Follow these steps to test your live installation:

1. **Upload a Research PDF**:
   - In the Streamlit sidebar under **Document Management**, click **Browse files** and select any PDF document.
   - Click the **Upload** button.
   - You will see chunking status and confirmation of vector indexing in ChromaDB.

2. **Run a RAG Chat Query**:
   - Navigate to the **💬 Research Chat** tab.
   - Ensure the **"Enable AI Evaluation"** toggle in the sidebar is turned **ON**.
   - Ask a question based on your uploaded paper (e.g., *"What is the main objective of this paper?"*).
   - The system retrieves context chunks, synthesizes a grounded answer, displays source citations with page numbers, and attaches the 5-dimension evaluation scorecard below the answer.

3. **Check Quality Analytics**:
   - Switch to the **📊 Analytics** tab.
   - View your session metrics: Total Queries, Average Quality Score, Evaluations Run, and the rolling trend line chart.

4. **Experiment in the Evaluation Lab**:
   - Switch to the **🧪 Evaluation Lab** tab.
   - Test custom Question-Answer-Context triples with or without Ground Truth to inspect how each evaluator scores edge cases, partial answers, or hallucinations.

---

## 🛠️ Troubleshooting & FAQs

### 1. `429 RESOURCE_EXHAUSTED` (Google Free Tier Rate Limit)
- **Cause**: Google Gemini Free Tier limits embedding requests to 100 per minute. Large research papers with hundreds of chunks can exhaust this quota when ingested too quickly.
- **Solution**: The engine is pre-equipped with an automated batching system (25 chunks per batch), 1-second pacing, and an intelligent backoff retry mechanism. If you ingest multi-hundred-page textbooks, allow the retry mechanism to finish pacing automatically.

### 2. `404 NOT_FOUND` for Embedding or Chat Model
- **Cause**: Deprecated or incorrect model name specified in `.env` (e.g. using `text-embedding-3-small` with `LLM_PROVIDER=google`).
- **Solution**:
  - For Google: Use `LLM_MODEL=gemini-3.5-flash-lite` and `EMBEDDING_MODEL=gemini-embedding-001`.
  - For OpenAI: Use `LLM_MODEL=gpt-4o-mini` and `EMBEDDING_MODEL=text-embedding-3-small`.

### 3. Port 8000 or 8501 Already in Use
- Change the FastAPI port:
  ```bash
  uvicorn app.api.routes:app --port 8080 --reload
  ```
- Change the Streamlit port:
  ```bash
  streamlit run streamlit_app.py --server.port 8502
  ```
  *(Note: Update the backend API URL in Streamlit sidebar settings if changing port 8000).*

### 4. Clearing the Vector Database
To reset the vector index and start fresh:
- In the Streamlit sidebar, click **"🗑️ Clear All Documents"**, OR
- Delete the local directory:
  ```bash
  rm -rf ./data/chroma_db    # Linux / macOS
  Remove-Item -Recurse -Force ./data/chroma_db  # Windows PowerShell
  ```

---
