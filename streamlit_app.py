"""
Streamlit frontend for the AI Research Analyzer RAG Chatbot.

Provides a full-featured web interface with:
- Document upload and management
- Research chat with RAG-powered responses
- Evaluation lab for manual testing
- Analytics dashboard
"""

import streamlit as st
import httpx
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300.0  # seconds – allows indexing large papers with rate limit backoff


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🔬 AI Research Analyzer",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        color: white;
        font-weight: 600;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card .score {
        font-size: 2rem;
        margin: 0.25rem 0;
        font-weight: 700;
    }
    .metric-card .label {
        font-size: 0.85rem;
        opacity: 0.95;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .metric-green  { background: linear-gradient(135deg, #2ecc71, #27ae60); }
    .metric-yellow { background: linear-gradient(135deg, #f1c40f, #f39c12); color: #2c3e50; }
    .metric-red    { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .metric-gray   { background: linear-gradient(135deg, #95a5a6, #7f8c8d); }
    .badge-pass {
        background-color: #2ecc71; color: white; padding: 0.25rem 0.75rem;
        border-radius: 1rem; font-weight: 700; font-size: 0.9rem;
    }
    .badge-warn {
        background-color: #f39c12; color: white; padding: 0.25rem 0.75rem;
        border-radius: 1rem; font-weight: 700; font-size: 0.9rem;
    }
    .badge-fail {
        background-color: #e74c3c; color: white; padding: 0.25rem 0.75rem;
        border-radius: 1rem; font-weight: 700; font-size: 0.9rem;
    }

    /* Ensure messages and cards have breathing room above the fixed bottom bar */
    .main .block-container {
        padding-bottom: 120px !important;
    }

    /* Keep the bottom docking bar clean and transparent */
    [data-testid="stBottom"] {
        background: transparent !important;
    }

    [data-testid="stBottom"] [data-testid="stChatInput"] {
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    /* Automatically hide the bottom chat bar when on Evaluation Lab or Analytics tabs */
    .stApp:has([data-baseweb="tab"]:nth-child(2)[aria-selected="true"]) [data-testid="stBottom"],
    .stApp:has([data-baseweb="tab"]:nth-child(3)[aria-selected="true"]) [data-testid="stBottom"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> httpx.Client:
    """Return a pre-configured synchronous httpx client."""
    return httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT)


def _score_color_class(score: float | None) -> str:
    """Return a CSS class name based on the score value."""
    if score is None:
        return "metric-gray"
    if score >= 0.8:
        return "metric-green"
    if score >= 0.5:
        return "metric-yellow"
    return "metric-red"


def _badge_html(verdict: str) -> str:
    """Return an HTML badge span for a verdict string."""
    key = str(verdict).upper()
    css = "badge-pass" if key == "PASS" else ("badge-warn" if key == "WARN" else "badge-fail")
    return f'<span class="{css}">{verdict}</span>'


def _render_metric_card(label: str, score: float | None) -> str:
    """Return HTML for a single coloured metric card."""
    if score is None:
        pct = "N/A"
        css_class = "metric-gray"
    else:
        pct = f"{score * 100:.0f}%"
        css_class = _score_color_class(score)

    return (
        f'<div class="metric-card {css_class}">'
        f'  <div class="label">{label}</div>'
        f'  <div class="score">{pct}</div>'
        f"</div>"
    )


def _api_get(path: str) -> dict[str, Any] | None:
    """Perform a GET request; return JSON or None on failure."""
    try:
        with _get_client() as client:
            resp = client.get(path)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        st.error("⚠️ Backend server is not running. Start with: `python main.py`")
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
    return None


def _api_post(path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Perform a POST request; return JSON or None on failure."""
    try:
        with _get_client() as client:
            resp = client.post(path, json=json)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        st.error("⚠️ Backend server is not running. Start with: `python main.py`")
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
    return None


def _api_delete(path: str) -> dict[str, Any] | None:
    """Perform a DELETE request; return JSON or None on failure."""
    try:
        with _get_client() as client:
            resp = client.delete(path)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text}")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        st.error("⚠️ Backend server is not running. Start with: `python main.py`")
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "enable_evaluation" not in st.session_state:
    st.session_state.enable_evaluation = True

if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if st.button("📤 Upload", use_container_width=True) and uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Uploading *{uf.name}*…"):
                try:
                    with _get_client() as client:
                        resp = client.post(
                            "/api/upload",
                            files={"file": (uf.name, uf.getvalue(), "application/pdf")},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        num = data.get("num_chunks") or data.get("chunks_created", "?")
                        st.success(f"✅ {uf.name} — {num} chunks indexed")
                except httpx.HTTPStatusError as exc:
                    st.error(f"❌ {uf.name}: {exc.response.text}")
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    st.error("⚠️ Backend server is not running. Start with: `python main.py`")
                except httpx.HTTPError as exc:
                    st.error(f"❌ {uf.name}: {exc}")

    st.divider()

    # ---- Settings ----------------------------------------------------------
    st.subheader("🔧 Settings")

    st.session_state.enable_evaluation = st.toggle(
        "Enable AI Evaluation",
        value=st.session_state.enable_evaluation,
    )

    if st.button("🗑️ Clear All Documents", use_container_width=True):
        result = _api_delete("/api/documents")
        if result is not None:
            st.success("All documents cleared from vector store.")

    st.divider()

    # ---- System health -----------------------------------------------------
    health = _api_get("/api/health")
    if health is not None:
        st.subheader("💚 System Health")
        st.metric("Status", health.get("status", "unknown").upper())
        st.metric("Documents", health.get("documents_count") or health.get("document_count", 0))
        st.metric("LLM Provider", health.get("llm_provider", "—"))
    else:
        st.caption("_Backend offline or unreachable._")


# ---------------------------------------------------------------------------
# Main area — Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_eval, tab_analytics = st.tabs(
    ["💬 Research Chat", "🧪 Evaluation Lab", "📊 Analytics"]
)

# ===========================  TAB 1: Research Chat  ========================
with tab_chat:
    st.subheader("💬 Research Chat")
    st.caption("Ask questions about your uploaded research papers. Responses are answered using RAG and automatically evaluated.")

    # Display chat history
    if not st.session_state.messages:
        st.info("👋 Welcome! Upload your research paper in the left sidebar and type your question below.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # If the assistant message carries evaluation data, render it
                if msg["role"] == "assistant" and msg.get("evaluation"):
                    _eval = msg["evaluation"]
                    with st.expander("📊 Evaluation Results"):
                        overall = _eval.get("overall_score", 0.0)
                        verdict = _eval.get("verdict", "N/A")
                        col_score, col_verdict = st.columns([1, 1])
                        with col_score:
                            st.markdown(
                                f"**Overall Score:** "
                                f'<span class="{_score_color_class(overall)}" '
                                f'style="padding:2px 8px;border-radius:6px;color:#fff;font-weight:700">'
                                f"{overall * 100:.1f}%</span>",
                                unsafe_allow_html=True,
                            )
                        with col_verdict:
                            st.markdown(f"**Verdict:** {_badge_html(verdict)}", unsafe_allow_html=True)

                        st.write("---")
                        metric_items = _eval.get("metrics") or _eval.get("scores", [])
                        for m in metric_items:
                            m_name = (m.get("name") or m.get("metric_name", "metric")).replace("_", " ").title()
                            m_score = m.get("score", 0.0)
                            st.caption(f"**{m_name}** ({m_score * 100:.0f}%)")
                            st.progress(min(max(float(m_score), 0.0), 1.0))
                            if m.get("reasoning"):
                                with st.expander(f"Reasoning – {m_name}"):
                                    st.write(m["reasoning"])

                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📄 Source Documents"):
                        for src in msg["sources"]:
                            fname = src.get("filename") or src.get("source", "Document")
                            page = src.get("page_number", src.get("page", 1))
                            chunk = src.get("chunk_id", src.get("chunk_index", "0"))
                            st.markdown(f"- 📄 **{fname}** (Page {page}, Chunk {chunk})")


# ========================  TAB 2: Evaluation Lab  =========================
with tab_eval:
    st.subheader("🧪 AI Evaluation Lab")
    st.caption("Benchmark any Question, Answer, and Context triple across all 5 evaluation metrics:")
    st.markdown("`Faithfulness` • `Answer Relevance` • `Context Relevance` • `Correctness` • `Completeness`")

    col_q, col_gt = st.columns(2)
    with col_q:
        eval_question = st.text_area(
            "Question",
            height=90,
            key="eval_q",
            placeholder="e.g. What is the impact of attention mechanisms on transformer models?",
        )
    with col_gt:
        eval_ground_truth = st.text_area(
            "Ground Truth (Optional - required for Correctness)",
            height=90,
            key="eval_gt",
            placeholder="e.g. Attention mechanisms allow models to focus on relevant input parts dynamically...",
        )

    eval_answer = st.text_area(
        "AI Answer to Evaluate",
        height=120,
        key="eval_a",
        placeholder="Enter the AI generated answer here...",
    )
    eval_context = st.text_area(
        "Retrieved Context Chunks (Separate chunks by newline or empty line)",
        height=140,
        key="eval_c",
        placeholder="Chunk 1: Attention mechanisms were introduced...\n\nChunk 2: In transformer architectures...",
    )

    if st.button("🚀 Run Full Evaluation Suite", use_container_width=True):
        if not eval_question.strip() or not eval_answer.strip():
            st.warning("Please provide at least a Question and an AI Answer.")
        else:
            context_chunks = [
                line.strip()
                for line in eval_context.split("\n\n")
                if line.strip()
            ]
            if not context_chunks:
                context_chunks = [
                    line.strip()
                    for line in eval_context.splitlines()
                    if line.strip()
                ]

            payload: dict[str, Any] = {
                "question": eval_question.strip(),
                "answer": eval_answer.strip(),
                "context": context_chunks if context_chunks else [eval_answer.strip()],
            }
            if eval_ground_truth.strip():
                payload["ground_truth"] = eval_ground_truth.strip()

            with st.spinner("Running 5 evaluation metrics in parallel…"):
                result = _api_post("/api/evaluate", json=payload)

            if result is not None:
                st.write("---")
                overall = result.get("overall_score", 0.0)
                verdict = result.get("verdict", "N/A")

                col_top1, col_top2 = st.columns([2, 1])
                with col_top1:
                    st.subheader(f"Overall Evaluation Verdict: {_badge_html(verdict)}")
                with col_top2:
                    st.markdown(
                        f"### Aggregate Score: **{overall * 100:.1f}%**"
                    )

                # Metric cards in columns
                metric_keys = [
                    ("faithfulness", "Faithfulness"),
                    ("answer_relevance", "Answer Relevance"),
                    ("context_relevance", "Context Relevance"),
                    ("correctness", "Correctness"),
                    ("completeness", "Completeness"),
                ]

                metric_scores_dict: dict[str, float | None] = {}
                metric_reasoning_dict: dict[str, str] = {}

                # Extract from scores / metrics list
                items = result.get("metrics") or result.get("scores", [])
                for it in items:
                    m_key = it.get("name") or it.get("metric_name", "")
                    metric_scores_dict[m_key] = it.get("score")
                    metric_reasoning_dict[m_key] = it.get("reasoning", "")

                cols = st.columns(len(metric_keys))
                for col, (k, title) in zip(cols, metric_keys):
                    score = metric_scores_dict.get(k)
                    with col:
                        st.markdown(
                            _render_metric_card(title, score),
                            unsafe_allow_html=True,
                        )

                # Detailed reasoning
                st.write("---")
                st.subheader("📝 Metric Details & Reasoning")
                for k, title in metric_keys:
                    reasoning = metric_reasoning_dict.get(k)
                    score = metric_scores_dict.get(k)
                    if reasoning:
                        score_display = f"{score * 100:.0f}%" if score is not None else "Skipped"
                        with st.expander(f"🔍 {title} ({score_display})", expanded=True):
                            st.write(reasoning)

                with st.expander("Raw Evaluation Payload"):
                    st.json(result)


# ==========================  TAB 3: Analytics  =============================
with tab_analytics:
    st.subheader("📊 Analytics & Quality Trends")
    st.caption("Track quality metrics across queries in the current session.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", st.session_state.query_count)
    with col2:
        if st.session_state.eval_scores:
            avg = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
            st.metric("Average Quality Score", f"{avg * 100:.1f}%")
        else:
            st.metric("Average Quality Score", "—")
    with col3:
        st.metric("Evaluations Run", len(st.session_state.eval_scores))

    if st.session_state.eval_scores:
        st.write("---")
        st.line_chart(
            st.session_state.eval_scores,
            y_label="Quality Score (0.0 - 1.0)",
        )
    else:
        st.info("Ask questions in the Research Chat tab with evaluation enabled to see quality trends here.")


# ===========================================================================
# Root-Level Fixed Bottom Chat Input
# Anchors natively to the bottom of the viewport (data-testid="stBottom")
# ===========================================================================
if prompt := st.chat_input("Ask about your research documents…"):
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call backend
    with st.spinner("Analyzing documents…"):
        payload = {
            "query": prompt,
            "question": prompt,
            "include_evaluation": st.session_state.enable_evaluation,
        }
        data = _api_post("/api/chat", json=payload)

    if data is not None:
        answer = data.get("answer", "_No answer returned._")
        eval_data = data.get("evaluation")
        if eval_data:
            st.session_state.eval_scores.append(eval_data.get("overall_score", 0.0))

        msg_record = {
            "role": "assistant",
            "content": answer,
            "sources": data.get("sources", []),
            "evaluation": eval_data,
        }
        st.session_state.query_count += 1
        st.session_state.messages.append(msg_record)
    else:
        fallback = "Sorry, I couldn't get a response. Is the backend server running?"
        st.session_state.messages.append(
            {"role": "assistant", "content": fallback}
        )

    st.rerun()

