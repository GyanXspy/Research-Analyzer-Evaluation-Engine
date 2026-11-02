"""
Comprehensive PDF Generator for AI Research Analyzer & Evaluation Engine
Creates a publication-quality, multi-page Technical Deep Dive & Interview Revision Guide.
"""

import os
import sys
import textwrap
import pymupdf

class InterviewGuideBuilder:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.doc = pymupdf.open()
        
        # Page metrics (A4)
        self.width = 595.0
        self.height = 842.0
        self.margin_x = 45.0
        self.margin_top = 45.0
        self.margin_bottom = 790.0
        self.content_width = self.width - (2 * self.margin_x) # 505 pt
        
        # Elegant Color Palette
        self.C_NAVY = (0.08, 0.18, 0.36)        # #142e5c - Primary headers
        self.C_TEAL = (0.05, 0.58, 0.55)        # #0e948c - Secondary accents
        self.C_DARK = (0.13, 0.15, 0.18)        # #212529 - Body text
        self.C_MUTED = (0.42, 0.45, 0.50)       # #6c757d - Captions/footers
        self.C_LIGHT_BG = (0.96, 0.97, 0.99)    # #f5f8fc - Hero/Table alternate
        self.C_CARD_BG = (0.98, 0.985, 0.995)   # #fafbfe - Question card background
        self.C_BORDER = (0.84, 0.88, 0.93)      # #d6e0ed - Borders
        self.C_SUCCESS = (0.12, 0.55, 0.25)     # #1f8c40 - Positive indicators
        self.C_HIGHLIGHT = (0.93, 0.96, 1.0)    # #edf5ff - Callout box background
        self.C_ACCENT_LINE = (0.20, 0.45, 0.75) # #3373bf - Callout border
        
        self.current_page = None
        self.cursor_y = self.margin_top
        self.new_page()

    def new_page(self):
        """Creates a new page with standard background and running headers."""
        self.current_page = self.doc.new_page(width=self.width, height=self.height)
        
        # Top primary teal accent line
        self.current_page.draw_rect(pymupdf.Rect(0, 0, self.width, 4), color=self.C_TEAL, fill=self.C_TEAL)
        
        # Running header (for pages > 1)
        if len(self.doc) > 1:
            header_rect = pymupdf.Rect(self.margin_x, 18, self.width - self.margin_x, 32)
            self.current_page.insert_textbox(
                header_rect,
                "AI Research Analyzer & Evaluation Engine  |  Technical & Interview Revision Guide",
                fontsize=8,
                fontname="helv",
                color=self.C_MUTED
            )
            self.current_page.draw_line(
                pymupdf.Point(self.margin_x, 34),
                pymupdf.Point(self.width - self.margin_x, 34),
                color=self.C_BORDER,
                width=0.6
            )
            self.cursor_y = 50.0
        else:
            self.cursor_y = self.margin_top

    def check_space(self, required_height: float):
        """Checks if enough vertical space is available; creates new page if not."""
        if self.cursor_y + required_height > self.margin_bottom:
            self.new_page()

    def estimate_text_height(self, text: str, font_size: float, width: float) -> float:
        """Estimates wrapped text height accurately."""
        chars_per_line = max(20, int(width / (font_size * 0.52)))
        lines = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            wrapped = textwrap.wrap(paragraph, width=chars_per_line)
            lines.extend(wrapped if wrapped else [""])
        line_height = font_size * 1.34
        return max(font_size + 4, len(lines) * line_height)

    def add_title_banner(self, title: str, subtitle: str, meta_items: list):
        """Renders an executive hero banner."""
        banner_height = 95.0
        self.check_space(banner_height + 15)
        
        rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + banner_height)
        self.current_page.draw_rect(rect, color=self.C_BORDER, fill=self.C_LIGHT_BG, width=0.8)
        
        # Left decorative vertical accent pill
        accent_rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.margin_x + 5, self.cursor_y + banner_height)
        self.current_page.draw_rect(accent_rect, color=self.C_NAVY, fill=self.C_NAVY)
        
        # Title
        t_rect = pymupdf.Rect(self.margin_x + 18, self.cursor_y + 12, self.width - self.margin_x - 15, self.cursor_y + 40)
        self.current_page.insert_textbox(t_rect, title, fontsize=16, fontname="helv", color=self.C_NAVY)
        
        # Subtitle
        s_rect = pymupdf.Rect(self.margin_x + 18, self.cursor_y + 42, self.width - self.margin_x - 15, self.cursor_y + 62)
        self.current_page.insert_textbox(s_rect, subtitle, fontsize=10.0, fontname="helv", color=self.C_TEAL)
        
        # Metadata items (pipe separated)
        meta_str = "  |  ".join(meta_items)
        m_rect = pymupdf.Rect(self.margin_x + 18, self.cursor_y + 68, self.width - self.margin_x - 15, self.cursor_y + 88)
        self.current_page.insert_textbox(m_rect, meta_str, fontsize=8.2, fontname="helv", color=self.C_MUTED)
        
        self.cursor_y += banner_height + 16.0

    def add_h1(self, text: str):
        """Major section heading (Level 1)."""
        self.check_space(38)
        self.cursor_y += 6
        
        # Underline accent line
        p1 = pymupdf.Point(self.margin_x, self.cursor_y + 20)
        p2 = pymupdf.Point(self.width - self.margin_x, self.cursor_y + 20)
        self.current_page.draw_line(p1, p2, color=self.C_BORDER, width=0.9)
        
        # Heading text with teal left pip
        pip = pymupdf.Rect(self.margin_x, self.cursor_y + 4, self.margin_x + 4, self.cursor_y + 17)
        self.current_page.draw_rect(pip, color=self.C_TEAL, fill=self.C_TEAL)
        
        t_rect = pymupdf.Rect(self.margin_x + 12, self.cursor_y, self.width - self.margin_x, self.cursor_y + 20)
        self.current_page.insert_textbox(t_rect, text.upper(), fontsize=11.5, fontname="helv", color=self.C_NAVY)
        
        self.cursor_y += 28.0

    def add_h2(self, text: str):
        """Sub-section heading (Level 2)."""
        self.check_space(24)
        self.cursor_y += 4
        
        t_rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + 16)
        self.current_page.insert_textbox(t_rect, text, fontsize=10.0, fontname="helv", color=self.C_TEAL)
        self.cursor_y += 18.0

    def add_paragraph(self, text: str, bold_prefix: str = "", font_size: float = 8.8, color=None, space_after: float = 6.0):
        """Inserts text with automatic line wrapping."""
        if color is None:
            color = self.C_DARK
            
        full_text = f"{bold_prefix}{text}" if bold_prefix else text
        req_height = self.estimate_text_height(full_text, font_size, self.content_width)
        
        self.check_space(req_height + space_after)
        
        rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + req_height + 8)
        self.current_page.insert_textbox(rect, full_text, fontsize=font_size, fontname="helv", color=color)
        self.cursor_y += req_height + space_after

    def add_bullet(self, text: str, bold_prefix: str = "", indent: float = 14.0):
        """Inserts bullet point with clean dot."""
        full_text = f"{bold_prefix} {text}" if bold_prefix else text
        eff_width = self.content_width - indent
        req_height = self.estimate_text_height(full_text, 8.6, eff_width)
        
        self.check_space(req_height + 4.0)
        
        # Draw bullet dot
        dot_center = pymupdf.Point(self.margin_x + 5, self.cursor_y + 6)
        self.current_page.draw_circle(dot_center, 1.8, color=self.C_TEAL, fill=self.C_TEAL)
        
        rect = pymupdf.Rect(self.margin_x + indent, self.cursor_y, self.width - self.margin_x, self.cursor_y + req_height + 6)
        self.current_page.insert_textbox(rect, full_text, fontsize=8.6, fontname="helv", color=self.C_DARK)
        self.cursor_y += req_height + 4.0

    def add_callout(self, text: str, title: str = "KEY TAKEAWAY", border_color=None, bg_color=None):
        """Draws a highlighted executive callout box."""
        if border_color is None:
            border_color = self.C_ACCENT_LINE
        if bg_color is None:
            bg_color = self.C_HIGHLIGHT
            
        full_text = f"[{title.upper()}]\n{text}"
        box_width = self.content_width - 24.0
        req_height = self.estimate_text_height(full_text, 8.6, box_width)
        total_height = req_height + 14.0
        
        self.check_space(total_height + 8.0)
        
        # Box background & border
        box_rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + total_height)
        self.current_page.draw_rect(box_rect, color=border_color, fill=bg_color, width=0.7)
        
        # Left thick bar
        accent_bar = pymupdf.Rect(self.margin_x, self.cursor_y, self.margin_x + 4, self.cursor_y + total_height)
        self.current_page.draw_rect(accent_bar, color=border_color, fill=border_color)
        
        # Text inside
        t_rect = pymupdf.Rect(self.margin_x + 14, self.cursor_y + 6, self.width - self.margin_x - 10, self.cursor_y + total_height - 4)
        self.current_page.insert_textbox(t_rect, full_text, fontsize=8.6, fontname="helv", color=self.C_NAVY)
        
        self.cursor_y += total_height + 8.0

    def add_table(self, headers: list, rows: list, col_proportions: list):
        """Draws an aligned, zebra-striped table."""
        col_widths = [p * self.content_width for p in col_proportions]
        row_height = 20.0
        header_height = 20.0
        total_height = header_height + (len(rows) * row_height)
        
        self.check_space(total_height + 10.0)
        
        start_y = self.cursor_y
        
        # Draw Header Row
        h_rect = pymupdf.Rect(self.margin_x, start_y, self.width - self.margin_x, start_y + header_height)
        self.current_page.draw_rect(h_rect, color=self.C_NAVY, fill=self.C_NAVY)
        
        cur_x = self.margin_x
        for i, h in enumerate(headers):
            cell_rect = pymupdf.Rect(cur_x + 4, start_y + 5, cur_x + col_widths[i] - 4, start_y + header_height - 2)
            self.current_page.insert_textbox(cell_rect, h, fontsize=8.2, fontname="helv", color=(1, 1, 1))
            cur_x += col_widths[i]
            
        cur_y = start_y + header_height
        
        # Draw Data Rows
        for r_idx, row in enumerate(rows):
            fill_c = self.C_LIGHT_BG if r_idx % 2 == 1 else (1, 1, 1)
            r_rect = pymupdf.Rect(self.margin_x, cur_y, self.width - self.margin_x, cur_y + row_height)
            self.current_page.draw_rect(r_rect, color=self.C_BORDER, fill=fill_c, width=0.5)
            
            cur_x = self.margin_x
            for c_idx, val in enumerate(row):
                cell_rect = pymupdf.Rect(cur_x + 4, cur_y + 4, cur_x + col_widths[c_idx] - 4, cur_y + row_height - 2)
                self.current_page.insert_textbox(cell_rect, str(val), fontsize=7.8, fontname="helv", color=self.C_DARK)
                cur_x += col_widths[c_idx]
            cur_y += row_height
            
        self.cursor_y = cur_y + 8.0

    def add_image(self, img_path: str, height: float, caption: str = ""):
        """Embeds an image centered with border and caption."""
        if not os.path.exists(img_path):
            return
            
        cap_height = 14.0 if caption else 0.0
        self.check_space(height + cap_height + 12.0)
        
        img_rect = pymupdf.Rect(self.margin_x + 10, self.cursor_y, self.width - self.margin_x - 10, self.cursor_y + height)
        self.current_page.draw_rect(img_rect, color=self.C_BORDER, fill=(0.05, 0.08, 0.12), width=0.8)
        self.current_page.insert_image(img_rect, filename=img_path)
        
        self.cursor_y += height + 4
        if caption:
            c_rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + 12)
            self.current_page.insert_textbox(c_rect, f"Figure: {caption}", fontsize=7.5, fontname="helv", color=self.C_MUTED)
            self.cursor_y += 12.0
        self.cursor_y += 6.0

    def add_qa(self, q_num: int, question: str, answer: str, category: str = ""):
        """Renders an interview question and detailed answer card."""
        q_text = f"Q{q_num}: {question}" + (f"  [{category}]" if category else "")
        a_text = f"Strong Candidate Response:\n{answer}"
        
        w = self.content_width - 16.0
        h_q = self.estimate_text_height(q_text, 9.0, w)
        h_a = self.estimate_text_height(a_text, 8.4, w)
        
        total_card = h_q + h_a + 22.0
        self.check_space(total_card + 6.0)
        
        # Outer Card Container
        card_rect = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + total_card)
        self.current_page.draw_rect(card_rect, color=self.C_BORDER, fill=self.C_CARD_BG, width=0.6)
        
        # Question Header Band
        q_bar = pymupdf.Rect(self.margin_x, self.cursor_y, self.width - self.margin_x, self.cursor_y + h_q + 8)
        self.current_page.draw_rect(q_bar, color=self.C_NAVY, fill=self.C_NAVY)
        
        q_rect = pymupdf.Rect(self.margin_x + 8, self.cursor_y + 4, self.width - self.margin_x - 8, self.cursor_y + h_q + 6)
        self.current_page.insert_textbox(q_rect, q_text, fontsize=8.8, fontname="helv", color=(1, 1, 1))
        
        # Answer text body
        a_rect = pymupdf.Rect(self.margin_x + 10, self.cursor_y + h_q + 12, self.width - self.margin_x - 8, self.cursor_y + total_card - 4)
        self.current_page.insert_textbox(a_rect, a_text, fontsize=8.2, fontname="helv", color=self.C_DARK)
        
        self.cursor_y += total_card + 7.0

    def finalize(self):
        """Pass 2: Stamps page numbering footer on all pages and saves."""
        total_pages = len(self.doc)
        for i in range(total_pages):
            page = self.doc[i]
            # Bottom divider line
            page.draw_line(
                pymupdf.Point(self.margin_x, self.margin_bottom + 8),
                pymupdf.Point(self.width - self.margin_x, self.margin_bottom + 8),
                color=self.C_BORDER,
                width=0.6
            )
            # Left footer note
            f_rect_l = pymupdf.Rect(self.margin_x, self.margin_bottom + 12, self.margin_x + 320, self.margin_bottom + 26)
            page.insert_textbox(
                f_rect_l,
                "AI Evaluation Engine  |  Technical Interview Mastery Guide",
                fontsize=7.8,
                fontname="helv",
                color=self.C_MUTED
            )
            # Right footer page number
            f_rect_r = pymupdf.Rect(self.width - self.margin_x - 120, self.margin_bottom + 12, self.width - self.margin_x, self.margin_bottom + 26)
            page.insert_textbox(
                f_rect_r,
                f"Page {i + 1} of {total_pages}",
                fontsize=8.0,
                fontname="helv",
                color=self.C_NAVY
            )
            
        self.doc.save(self.output_path)
        self.doc.close()
        print(f"Master PDF successfully generated: {self.output_path} ({total_pages} pages)")


def build_full_interview_pdf():
    pdf_path = os.path.abspath("AI_Research_Analyzer_Evaluation_Engine_Interview_Mastery_Guide.pdf")
    builder = InterviewGuideBuilder(pdf_path)
    
    # ---------------------------------------------------------
    # COVER / HERO
    # ---------------------------------------------------------
    builder.add_title_banner(
        title="AI Research Analyzer & Evaluation Engine",
        subtitle="Complete Technical Deep Dive, System Architecture & Interview Revision Guide",
        meta_items=[
            "Target Role: Generative AI / LLM Engineer",
            "Tech Stack: LangChain • LangGraph • ChromaDB • FastAPI • Streamlit",
            "Live Benchmark: 90.5% Average Quality Score (30 Queries)"
        ]
    )
    
    # ---------------------------------------------------------
    # SECTION 1: EXECUTIVE OVERVIEW & STAR PITCH
    # ---------------------------------------------------------
    builder.add_h1("1. Executive Summary & 30-Second Elevator Pitch")
    
    builder.add_callout(
        text=(
            "\"I built an enterprise-grade AI Research Assistant and automated Evaluation Engine that ingests dense "
            "academic documents, executes stateful semantic retrieval via LangGraph, and mitigates hallucinations in real-time "
            "across 5 quality dimensions (Faithfulness, Answer Relevance, Context Relevance, Semantic Correctness, and Completeness). "
            "In production testing across 30 live queries, the engine maintained a 90.5% average quality score, achieved a 100% "
            "evaluation capture rate, and resolved Google API 429 quota exhaustion through intelligent micro-batching and exponential backoff.\""
        ),
        title="THE 30-SECOND ELEVATOR PITCH (STAR FORMAT)"
    )
    
    builder.add_h2("The Core Problem: Silent Failures in Enterprise RAG")
    builder.add_paragraph(
        "Standard Retrieval-Augmented Generation (RAG) systems suffer from three primary silent failure modes that cannot be caught by standard HTTP status codes:"
    )
    builder.add_bullet(
        "Context Noise / Retriever Drift: The embedding model pulls irrelevant or tangent chunks that clutter the LLM context window.",
        "1. Noisy Retrieval:"
    )
    builder.add_bullet(
        "Extrinsic Hallucinations: The generator synthesizes facts not grounded in the source text, presenting unverified claims with high confidence.",
        "2. Hallucinations:"
    )
    builder.add_bullet(
        "Critical Information Omission: The LLM answers superficially, omitting essential caveats, formulas, or statistical findings present in the source.",
        "3. Incompleteness:"
    )
    
    builder.add_paragraph(
        "The AI Evaluation Engine resolves this by running a parallel LLM-as-a-Judge inspection pipeline for every query, producing granular 0.0-1.0 confidence scorecards and an overall PASS/FAIL verdict before responses reach downstream users."
    )

    # ---------------------------------------------------------
    # SECTION 2: END-TO-END ARCHITECTURE
    # ---------------------------------------------------------
    builder.add_h1("2. High-Level & Component Architecture")
    
    builder.add_h2("The Three Core Pipelines")
    builder.add_table(
        headers=["Pipeline Stage", "Primary Technologies", "Key Engineering Decisions & Rationale"],
        rows=[
            ["1. Ingestion & Indexing", "PyMuPDF, RecursiveSplitter, ChromaDB", "Chunk size 1000 with 200 overlap preserves semantic paragraph coherence. Micro-batching (25/batch) prevents 429 quota limits."],
            ["2. RAG Graph Workflow", "LangGraph StateGraph, Gemini / OpenAI", "Stateful graph with explicit nodes ('retrieve', 'generate', 'format_sources'). Eliminates brittle LCEL linear piping."],
            ["3. Evaluation Engine", "LLM-as-a-Judge, Pydantic v2, Async Engine", "5 independent metric evaluators execute in parallel. JSON output validation guarantees deterministic parsing."]
        ],
        col_proportions=[0.24, 0.32, 0.44]
    )
    
    builder.add_h2("Deep Dive: Document Ingestion & Chunking")
    builder.add_bullet(
        "Why PyMuPDF (fitz)? 15-20x faster extraction than PyPDF2, robust multi-column layout support, and accurate page number extraction.",
        "• Parsing Engine:"
    )
    builder.add_bullet(
        "chunk_size=1000 chars (~200-250 tokens), chunk_overlap=200 chars. Rationale: Academic paper paragraphs typically span 600-900 chars; 1000 chars captures complete ideas without truncation, while 200 overlap ensures boundary sentences are never lost.",
        "• Chunking Hyperparameters:"
    )
    builder.add_bullet(
        "ChromaDB with cosine distance metric and persistent disk storage (./data/chroma_db). Preserves metadata: {source: filename, page_number: int, chunk_index: int}.",
        "• Vector Storage:"
    )

    builder.add_h2("LangGraph StateGraph Workflow vs Linear Chains")
    builder.add_paragraph(
        "Traditional LangChain LCEL chains execute as opaque black boxes. In this project, the RAG cycle is modeled as a state machine using LangGraph:"
    )
    builder.add_bullet(
        "Explicit State Definition: RAGState tracks question, context list, answer, and source_documents across all steps.",
        "1. State Transparency:"
    )
    builder.add_bullet(
        "Node 'retrieve': Fetches top-k=4 semantic chunks from ChromaDB with similarity thresholds.",
        "2. Retrieval Node:"
    )
    builder.add_bullet(
        "Node 'generate': Context-conditioned prompt with zero-shot grounding rules instructs the model to only use provided context.",
        "3. Synthesis Node:"
    )
    builder.add_bullet(
        "Node 'format_sources': Deduplicates cited documents, validates page numbers, and constructs clean reference metadata.",
        "4. Formatting Node:"
    )

    # ---------------------------------------------------------
    # SECTION 3: THE 5-PILLAR EVALUATION ENGINE
    # ---------------------------------------------------------
    builder.add_h1("3. The 5-Pillar Evaluation Framework")
    
    builder.add_paragraph(
        "The evaluation engine operates under an LLM-as-a-Judge paradigm, assessing answers across both Reference-Free (production real-time) and Reference-Based (lab testing) dimensions:"
    )
    
    builder.add_table(
        headers=["Metric Name", "Evaluator Class", "Ref?", "What It Measures & Underlying Algorithm"],
        rows=[
            ["Faithfulness", "FaithfulnessEvaluator", "No", "Deconstructs generated answer into atomic claims; checks whether every claim logically entails from retrieved context. Detects hallucinations."],
            ["Answer Relevance", "AnswerRelevanceEvaluator", "No", "Analyzes semantic alignment between query intent and answer content. Penalizes evasive, non-responsive, or off-topic text."],
            ["Context Relevance", "ContextRelevanceEvaluator", "No", "Measures retrieval precision by evaluating each context chunk individually. Penalizes noisy, irrelevant chunk retrieval."],
            ["Correctness", "CorrectnessEvaluator", "Yes", "Compares answer against authoritative ground-truth benchmark. Evaluates semantic equivalence, not mere keyword matching."],
            ["Completeness", "CompletenessEvaluator", "No", "Audits whether all critical facts, mechanisms, and caveats present in retrieved context were included in the response."]
        ],
        col_proportions=[0.20, 0.26, 0.08, 0.46]
    )
    
    builder.add_h2("Scoring Formula & Verdict Threshold")
    builder.add_paragraph(
        "The Overall Quality Score is the unweighted arithmetic mean of all executed metric evaluators:"
    )
    builder.add_callout(
        text=(
            "Overall Score = (1 / N) * Sum(Score_i)   for i in [1..N]\n"
            "Threshold Policy: If Overall Score >= 0.70  ==>  VERDICT = PASS\n"
            "                  If Overall Score <  0.70  ==>  VERDICT = FAIL\n"
            "Guarantees deterministic, machine-readable quality gates in CI/CD and production monitoring."
        ),
        title="MATHEMATICAL FORMULA & VERDICT POLICY"
    )

    # ---------------------------------------------------------
    # SECTION 4: EMPIRICAL BENCHMARKS & CASE STUDY
    # ---------------------------------------------------------
    builder.add_h1("4. Real-World Benchmarks & Evaluation Analytics")
    
    builder.add_paragraph(
        "The engine was rigorously validated in an interactive 30-query testing session querying dense, multi-page research documents (e.g. Distributed Ray Systems, AGI Concept Foundations). Below are the empirical results:"
    )
    
    builder.add_image("assets/evaluation_metrics.png", height=125.0, caption="Live 30-Query Quality Score Trend (Average: 90.5%)")
    
    builder.add_h2("Key Benchmark Takeaways:")
    builder.add_bullet(
        "Total Queries: 30  |  Evaluations Run: 30  |  Average Quality Score: 90.5% (0.905 / 1.0).",
        "• High Reliability:"
    )
    builder.add_bullet(
        "Over 90% of queries consistently scored between 0.85 and 1.0, proving strong grounding and high retrieval precision.",
        "• Consistency:"
    )
    builder.add_bullet(
        "At query #6, the score dropped sharply to 0.40. Cause: The retriever pulled introduction background chunks instead of the specific architectural component asked. The evaluation engine immediately flagged this as Context Noise & Partial Hallucination, demonstrating sensitive anomaly detection.",
        "• Anomaly Detection (Query #6):"
    )
    
    builder.add_image("assets/chat_interface.png", height=120.0, caption="Streamlit Research Chatbot Interface with Live Citation Display")

    # ---------------------------------------------------------
    # SECTION 5: PRODUCTION RESILIENCE & PROBLEM SOLVING
    # ---------------------------------------------------------
    builder.add_h1("5. Production Engineering & Bug Triage Case Studies")
    
    builder.add_h2("Case Study 1: Resolving Google 429 RESOURCE_EXHAUSTED Quota Limits")
    builder.add_paragraph(
        "Challenge: Ingesting dense 30-50 page research papers generated 150-250 chunks. Google Gemini's Free Tier strictly enforces 100 embedContent requests per minute, crashing the upload pipeline with 429 errors."
    )
    builder.add_paragraph(
        "Solution: Implemented a 3-tier resilience architecture in app/services/vector_store.py:"
    )
    builder.add_bullet("1. Micro-Batching: Chunk ingestion split into slices of 25 chunks per API call.")
    builder.add_bullet("2. Inter-Batch Pacing: A polite 1.0-second delay between batches to respect request burst limits.")
    builder.add_bullet("3. Regex-Driven Adaptive Backoff: If a 429 occurs, regex extracts retry delay from server headers (e.g. retryDelay: 15.2s) and sleeps exactly that duration before retrying.")

    builder.add_h2("Case Study 2: Pydantic v2 Type Coercion for DocumentMetadata")
    builder.add_paragraph(
        "Challenge: Streamlit chat crashed with 'ValidationError: 1 validation error for DocumentMetadata, page_number: Input should be a valid integer, input_value=None'."
    )
    builder.add_paragraph(
        "Solution: Built an input pre-validator in app/models/schemas.py with mode='before':"
    )
    builder.add_bullet("Automatically coerces None, missing values, or non-digit strings to 0 or int(value) prior to model instantiation.")

    # ---------------------------------------------------------
    # SECTION 6: 25+ COMPREHENSIVE INTERVIEW QUESTIONS & ANSWERS
    # ---------------------------------------------------------
    builder.add_h1("6. Comprehensive Interview Questions & Detailed Answers")
    
    # Category A: System Design
    builder.add_h2("Category A: Architecture & System Design")
    
    builder.add_qa(
        1,
        "Why did you choose LangGraph over standard LangChain sequential chains?",
        "LangChain's legacy sequential chains (and basic LCEL pipes) assume a purely linear, unbranching execution flow. "
        "In production RAG, workflows are non-linear: we need state inspection, conditional fallbacks (e.g., if retrieved context score is below threshold, re-retrieve or rewrite query), "
        "and clean state passing across retrieval, generation, and source formatting. LangGraph models the application as a formal StateGraph, providing cyclical graph support, "
        "transparent inspection of intermediate state keys, and production-grade reliability.",
        "System Design"
    )
    
    builder.add_qa(
        2,
        "How did you choose chunk size 1000 and overlap 200? What are the tradeoffs?",
        "In academic PDFs, an average dense technical paragraph consists of 600-900 characters. "
        "A chunk size of 1000 characters ensures that single cohesive concepts, algorithms, or definitions fit within a single chunk without semantic fragmentation. "
        "The 200-character overlap (~2-3 sentences) guarantees that boundary context (such as pronouns referring to the previous sentence) is preserved across chunks. "
        "Tradeoff: Smaller chunks (e.g., 256) increase retrieval precision but lose macro context. Larger chunks (e.g., 2048) capture broad context but dilute embedding similarity scores.",
        "RAG Engineering"
    )

    builder.add_qa(
        3,
        "Why use ChromaDB instead of cloud-managed alternatives like Pinecone or Weaviate?",
        "ChromaDB is lightweight, open-source, serverless, and runs in-process with local disk persistence (DuckDB/SQLite backend). "
        "For research document workloads where data privacy, zero recurring cloud infrastructure costs, and offline reproducibility are paramount, "
        "ChromaDB is ideal. In an enterprise production scale (>10M vectors), I would migrate to Pinecone, Milvus, or Qdrant with sharded clusters.",
        "Vector Databases"
    )

    builder.add_qa(
        4,
        "How would you scale this architecture to support 10,000 concurrent users?",
        "1. Decouple Ingestion & Retrieval: Move document chunking and embedding to an asynchronous Celery/Redis task queue.\n"
        "2. Distributed Vector Store: Migrate ChromaDB to an enterprise vector cluster (e.g., Qdrant or Milvus) with horizontal read replicas.\n"
        "3. Semantic Caching: Introduce Redis or GPTCache to cache embeddings and responses for frequent queries, reducing LLM calls by 30-40%.\n"
        "4. Asynchronous Evaluation: Decouple evaluation into an event-driven background pipeline (Kafka/SQS) so users receive responses in <1.5s while evaluation metrics are stored in ClickHouse/Prometheus.",
        "Scalability"
    )

    builder.add_qa(
        5,
        "How do you prevent context poisoning or prompt injection from malicious uploaded PDFs?",
        "1. Strict System Prompt Sandboxing: Delimit retrieved context using XML/markdown tags (e.g., <context>...</context>) and explicitly instruct the LLM to treat content inside context tags strictly as data, never as system instructions.\n"
        "2. Input Sanitization: Strip dangerous control characters, embedded scripts, and anomalous hidden text during PyMuPDF parsing.\n"
        "3. Faithfulness Evaluator as Security Gate: Prompt injection attacks typically coerce the LLM to ignore context; this causes the Faithfulness Evaluator score to drop to 0.0, automatically triggering a FAIL verdict.",
        "Security"
    )

    # Category B: Retrieval Mechanics
    builder.add_h2("Category B: RAG & Retrieval Mechanics")

    builder.add_qa(
        6,
        "How does dense vector retrieval work, and what metric did you use?",
        "Dense retrieval maps text chunks into continuous dense vector representations (e.g., 768 or 1536 dimensions) using an embedding model. "
        "Semantic similarity is computed using Cosine Similarity: cos(theta) = (A . B) / (||A|| * ||B||). "
        "Chunks with the highest cosine score to the user's query vector are returned. Unlike sparse BM25 keyword matching, dense retrieval captures semantic synonyms, intent, and paraphrased concepts.",
        "Retrieval"
    )

    builder.add_qa(
        7,
        "What is the 'Lost in the Middle' problem, and how do you address it?",
        "Research (Liu et al., 2023) demonstrates that LLMs attend best to information placed at the very beginning and very end of the prompt context window, "
        "frequently overlooking information buried in the middle. We mitigate this by: (1) Limiting k=4 top chunks to keep total context compact (~1000 tokens), "
        "and (2) Sorting retrieved chunks such that the highest similarity chunks are placed at the prompt boundaries.",
        "Prompt Engineering"
    )

    builder.add_qa(
        8,
        "What is the difference between Naive, Advanced, and Modular/Agentic RAG? Where does your project fit?",
        "• Naive RAG: Basic split -> embed -> vector search -> generate. Brittle and error-prone.\n"
        "• Advanced RAG: Adds pre-retrieval (query expansion, routing) and post-retrieval (reranking, contextual compression).\n"
        "• Modular / Agentic RAG: Uses state machines (LangGraph) for dynamic routing, tool use, and automated self-reflection/evaluation loops.\n"
        "Our system is a Modular Agentic RAG: It utilizes LangGraph for stateful execution, dynamically extracts metadata citations, and closes the loop with automated 5-pillar evaluation.",
        "RAG Taxonomy"
    )

    builder.add_qa(
        9,
        "How do you handle multi-column research papers and complex tables?",
        "Standard PDF parsers concatenate text horizontally across columns, corrupting sentence structure. "
        "We utilize PyMuPDF (`fitz`), which reads bounding box coordinates and extracts text column-by-column based on reading order. "
        "For tabular data, our chunking pipeline identifies table delimiters and preserves row-column integrity without breaking numeric relationships.",
        "Document Parsing"
    )

    # Category C: Evaluation & LLM-as-a-Judge
    builder.add_h2("Category C: Evaluation Engine & LLM-as-a-Judge")

    builder.add_qa(
        10,
        "How do you evaluate RAG quality when real users don't provide Ground Truth answers?",
        "By utilizing Reference-Free Evaluation Metrics. Out of our 5 dimensions, 4 do NOT require ground truth:\n"
        "1. Faithfulness: Verifies claims against retrieved context.\n"
        "2. Answer Relevance: Evaluates answer alignment with the user's question.\n"
        "3. Context Relevance: Assesses whether retrieved chunks match the query.\n"
        "4. Completeness: Audits whether the answer omitted facts present in context.\n"
        "Correctness is the only metric requiring ground truth, and is selectively activated during offline benchmark / gold dataset evaluations.",
        "Evaluation Theory"
    )

    builder.add_qa(
        11,
        "What are the known biases of LLM-as-a-Judge, and how do you mitigate them?",
        "1. Position Bias: Tendency to favor the first or last candidate in pairwise comparison. Mitigation: We use single-answer absolute scoring rubrics rather than pairwise comparisons.\n"
        "2. Verbosity Bias: Tendency to award higher scores to longer responses. Mitigation: Evaluators explicitly penalize extraneous text and assess atomic factual claims.\n"
        "3. Self-Enhancement Bias: Tendency to score outputs from the same model family higher. Mitigation: Use neutral, rubric-grounded system prompts with temperature=0.0 and structured JSON extraction.",
        "LLM Evaluation"
    )

    builder.add_qa(
        12,
        "Explain the Faithfulness Evaluator step-by-step.",
        "Step 1: The evaluator LLM parses the generated answer and decomposes it into discrete atomic claims (e.g., 'Claim 1: Ray uses a dynamic task execution engine').\n"
        "Step 2: For each atomic claim, the evaluator verifies if it logically entails from the retrieved context chunks (True / False / Unsupported).\n"
        "Step 3: Faithfulness Score = (Number of Entailed Claims) / (Total Number of Claims).\n"
        "If the answer includes unverified facts, the score decreases proportionately. A score < 0.70 flags a hallucination.",
        "Faithfulness Deep Dive"
    )

    builder.add_qa(
        13,
        "In your benchmark graph, query #6 dropped to 0.40. What happened, and why is this good?",
        "Query #6 asked: 'How does Ray achieve bottom-up distributed scheduling?'. "
        "The retriever returned general introductory chunks about Ray rather than Section 3.2 on local schedulers. "
        "As a result: (1) Context Relevance dropped to 0.35, (2) Faithfulness dropped to 0.50 because the model attempted to answer from general parametric knowledge, and (3) Completeness dropped to 0.30. "
        "This caused the Overall Score to drop to 0.40 (FAIL). This demonstrates that our evaluation engine is not a rubber stamp—it accurately detects retriever drift and protects users from hallucinated explanations.",
        "Empirical Validation"
    )

    builder.add_qa(
        14,
        "Why build a custom evaluation engine instead of using off-the-shelf Ragas or TruLens?",
        "Off-the-shelf libraries are heavy dependencies with frequent breaking API changes, black-box prompts, and rigid LLM client wrappers that break when new provider versions release (e.g. Google GenAI v1beta vs v1). "
        "Building a modular evaluation engine gave us:\n"
        "1. Full control over the scoring rubrics and claim decomposition algorithms.\n"
        "2. Native integration with our Pydantic v2 schemas and FastAPI response payloads.\n"
        "3. Direct error handling, retry backoff, and zero external dependency rot.",
        "Engineering Decision"
    )

    # Category D: Production Engineering & Behavioral
    builder.add_h2("Category D: Production Engineering & Behavioral Scenarios")

    builder.add_qa(
        15,
        "Describe a difficult bug you solved during this project.",
        "During large PDF ingestion, Google Gemini's Free Tier threw 429 RESOURCE_EXHAUSTED errors because embedding 200+ chunks exceeded the 100 requests/minute quota. "
        "Rather than artificially truncating documents, I re-architected the ingestion engine: (1) implemented 25-chunk micro-batching, (2) introduced 1-second inter-batch throttling, "
        "and (3) added an adaptive backoff retry loop using regular expressions to parse server delay recommendations (`retryDelay: 15s`). "
        "This completely eradicated upload crashes and allowed the system to ingest full 50-page textbooks flawlessly.",
        "Problem Solving"
    )

    builder.add_qa(
        16,
        "How do you test this project? What is your testing philosophy?",
        "We enforce a strict 16-test automated pytest suite covering:\n"
        "1. Configuration loading & provider switching (Google Gemini <-> OpenAI).\n"
        "2. Pydantic v2 schema validations and type coercions (handling null page numbers).\n"
        "3. Unit tests for each of the 5 evaluator metrics (mocking LLM judgments to verify math and edge cases).\n"
        "4. FastAPI endpoint integration tests using TestClient (/health, /upload, /chat, /evaluate).\n"
        "All 16 tests pass deterministically in under 3 seconds in CI/CD.",
        "Testing & CI/CD"
    )

    builder.add_qa(
        17,
        "What are the next three features you would build given another sprint?",
        "1. Hybrid Retrieval (BM25 + Dense Vectors) with Reciprocal Rank Fusion (RRF) to capture rare acronyms and exact code symbols.\n"
        "2. Cross-Encoder Reranker (e.g., Cohere Rerank or BGE-Reranker-Large) to reorder top-20 retrieved candidates down to top-4.\n"
        "3. Self-Correction / Corrective RAG (CRAG): If the evaluation engine outputs a FAIL verdict (score < 0.70), automatically rewrite the query and re-retrieve context before delivering the answer to the user.",
        "Product Roadmap"
    )

    # ---------------------------------------------------------
    # SECTION 7: INTERVIEW CHEAT SHEET & KEY TERMINOLOGY
    # ---------------------------------------------------------
    builder.add_h1("7. Quick-Fire Revision Cheat Sheet")
    
    builder.add_table(
        headers=["Term / Concept", "One-Line Technical Definition & Project Context"],
        rows=[
            ["Hallucination (Extrinsic)", "Generation of plausible-sounding facts unsupported by provided context. Caught by FaithfulnessEvaluator."],
            ["Context Noise", "Retrieved chunks that match keywords but lack answer signal. Caught by ContextRelevanceEvaluator."],
            ["LangGraph StateGraph", "Cyclic, stateful computational graph framework used to manage RAG execution stages."],
            ["Pydantic v2 Coercion", "Pre-validation logic ensuring malformed vector metadata does not trigger 500 server crashes."],
            ["Micro-Batching", "Grouping chunks into 25-item slices to obey Google Gemini free tier 100 req/min quotas."],
            ["LLM-as-a-Judge", "Using a foundation model to score model outputs against strict multi-criteria rubrics."],
            ["Cosine Distance", "Distance metric 1 - (A.B)/(||A||||B||) used by ChromaDB for nearest neighbor search."]
        ],
        col_proportions=[0.28, 0.72]
    )
    
    builder.add_callout(
        text=(
            "1. Lead with practical numbers: '30 live queries, 90.5% quality score, 5-pillar evaluation'.\n"
            "2. Emphasize why evaluation matters: 'Generative AI without evaluation is a liability in enterprise production.'\n"
            "3. Speak to tradeoffs: 'Why chunk size 1000? Why LangGraph over LCEL? Why ChromaDB over Pinecone?'\n"
            "4. Highlight resilience: Talk passionately about the 429 quota fix and Pydantic validation coercion."
        ),
        title="FINAL INTERVIEW ADVICE & MINDSET"
    )

    builder.finalize()
    return pdf_path

if __name__ == "__main__":
    build_full_interview_pdf()
