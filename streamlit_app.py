"""
AskTube — YouTube Video Q&A RAG Application
Streamlit Community Cloud Deployment Entry Point (₹0 / $0 Cost)

Reuses the core LangChain + RAG + FAISS + E5 Embeddings + Qwen3-8B pipeline
from backend.services.rag_service without duplicating RAG logic.
"""

import os
import sys
from typing import Optional, List
import streamlit as st

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import main
from backend.services.rag_service import rag_service
from backend.config import (
    MAX_QUESTION_LENGTH,
    EMBEDDING_MODEL,
    LLM_REPO_ID,
    RETRIEVER_K,
    RELEVANCE_SCORE_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AskTube — YouTube Video Q&A",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Polished AI Application Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
        margin: 0;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.75rem 0 1.25rem 0;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 500;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.25);
        color: #bbb;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.35);
        color: #10b981;
    }

    .video-card {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.25rem;
    }
    .video-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .video-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.85rem;
        color: #888;
    }

    .qa-box {
        padding: 1.25rem;
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(128, 128, 128, 0.18);
        margin-bottom: 1rem;
    }
    .question-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .answer-body {
        font-size: 0.98rem;
        line-height: 1.65;
        margin-bottom: 0.75rem;
    }

    .source-tree {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background: rgba(0, 0, 0, 0.25);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #38bdf8;
        margin-top: 0.5rem;
    }
    .source-item {
        margin-bottom: 0.5rem;
        line-height: 1.45;
    }
    .source-timestamp {
        font-weight: 600;
        color: #38bdf8;
    }
    .source-snippet {
        color: #aaa;
        margin-top: 0.2rem;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API Key / Secrets Resolution (₹0 Zero-Cost Verification)
# ---------------------------------------------------------------------------
def resolve_hf_token() -> Optional[str]:
    """Resolves Hugging Face API token with priority:
    1. Streamlit Secrets (for Community Cloud)
    2. Environment Variable (.env or system)
    3. User Sidebar Input Override
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
        except Exception:
            token = ""
    return token.strip() if token else None


# ---------------------------------------------------------------------------
# Cached Resource Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing multilingual embedding model...")
def get_cached_embeddings():
    """Initializes and caches the local E5 embedding model in memory."""
    return main.get_embeddings()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def get_youtube_timestamp_url(video_id: str, start_seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "current_video" not in st.session_state:
    st.session_state["current_video"] = None
    # Schema: {
    #     "url": str,
    #     "video_id": str,
    #     "title": str,
    #     "language": str,
    #     "is_generated": bool,
    #     "cached": bool,
    # }

if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []
    # Schema list of: {
    #     "question": str,
    #     "answer": str,
    #     "relevant_context_found": bool,
    #     "retrieved_sections": list,
    #     "status": str,
    # }


# ---------------------------------------------------------------------------
# Sidebar: System Status & Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System & Architecture")
    
    # Check current token
    current_token = resolve_hf_token()
    token_status = "✅ Connected" if current_token else "⚠️ Missing"
    
    st.markdown(
        f"""
        **LLM Provider:** Hugging Face Serverless  
        **Model:** `{LLM_REPO_ID}`  
        **Inference Cost:** ₹0 / $0 Free Tier  
        **Embeddings:** `{EMBEDDING_MODEL.split('/')[-1]}`  
        **Vector Store:** FAISS (Local CPU Cosine)  
        **Token Status:** {token_status}
        """
    )
    
    # Allow user override for token (useful for public portfolio viewers)
    with st.expander("🔑 Hugging Face API Token (Optional Override)", expanded=not bool(current_token)):
        st.caption(
            "AskTube uses free-tier Hugging Face serverless inference. If running without preset secrets, enter a free read token from huggingface.co/settings/tokens."
        )
        user_input_token = st.text_input(
            "API Token",
            value=current_token or "",
            type="password",
            placeholder="hf_...",
            help="Free read-only Hugging Face access token",
        )
        if user_input_token and user_input_token != os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = user_input_token.strip()
            # Reset singleton so model reloads with new credentials
            main._model_instance = None
            st.success("API token updated for this session.")

    st.markdown("---")
    
    # Active Video Details
    if st.session_state["current_video"]:
        cv = st.session_state["current_video"]
        st.markdown("### 📺 Active Video")
        st.write(f"**Title:** {cv['title']}")
        st.write(f"**ID:** `{cv['video_id']}`")
        st.write(f"**Language:** `{cv['language']}`")
        caption_type = "Auto-generated" if cv.get("is_generated") else "Manual/Standard"
        st.write(f"**Captions:** {caption_type}")
        st.write(f"**Index:** {'Loaded from Cache' if cv.get('cached') else 'Newly Embedded'}")
        
        if st.button("🔄 Clear / Load New Video", use_container_width=True):
            st.session_state["current_video"] = None
            st.session_state["qa_history"] = []
            st.rerun()
            
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #888;">
        <b>AskTube RAG Core</b><br>
        • Chunk Size: 1000 chars (200 overlap)<br>
        • Cosine Distance Threshold: 0.50<br>
        • Top-K Chunks: 4<br>
        • Zero-cost deployment on Streamlit Cloud
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>▶️ AskTube</h1>
        <p>Ask questions about any YouTube video using Retrieval-Augmented Generation (RAG).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Badges
st.markdown(
    """
    <div class="badge-container">
        <span class="badge badge-success">LangChain RAG</span>
        <span class="badge">E5 Multilingual Embeddings</span>
        <span class="badge">FAISS Vector Search</span>
        <span class="badge">Qwen3-8B</span>
        <span class="badge">Timestamp-Aware Citations</span>
        <span class="badge">100% Free Hosting</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Step 1: Video Input & Processing
# ---------------------------------------------------------------------------
input_col, btn_col = st.columns([4, 1])

with input_col:
    default_url = (
        st.session_state["current_video"]["url"]
        if st.session_state["current_video"]
        else ""
    )
    video_url = st.text_input(
        "YouTube Video URL",
        value=default_url,
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Paste a standard YouTube, Shorts, or Embed URL with captions.",
        label_visibility="collapsed",
    )

with btn_col:
    process_btn = st.button("Process Video", type="primary", use_container_width=True)

# Language option dropdown (optional)
with st.expander("🌐 Advanced Options (Language & Refresh)", expanded=False):
    col_lang, col_refresh = st.columns(2)
    with col_lang:
        lang_preference = st.selectbox(
            "Preferred Transcript Language",
            options=["en", "hi", "es", "fr", "de", "ja", "zh", "auto"],
            index=0,
            help="English is preferred by default, but AskTube will fall back to other available transcripts.",
        )
    with col_refresh:
        force_refresh = st.checkbox(
            "Force re-fetch transcript & re-build index",
            value=False,
            help="Bypasses local cache to re-index the video.",
        )

# Process Video Action
if process_btn:
    raw_url = (video_url or "").strip()
    if not raw_url:
        st.error("Please enter a valid YouTube video URL.")
    else:
        with st.status("Processing YouTube video...", expanded=True) as status:
            st.write("🔍 Validating YouTube URL...")
            try:
                vid = main.extract_video_id(raw_url)
                st.write(f"✓ Video ID extracted: `{vid}`")
            except Exception as e:
                status.update(label="Invalid URL", state="error")
                st.error(f"Invalid YouTube URL: {e}")
                st.stop()

            st.write("📥 Fetching transcript & subtitle metadata...")
            pref_lang = None if lang_preference == "auto" else lang_preference
            
            st.write("⚙️ Preparing chunks & checking FAISS index...")
            resp = rag_service.process_video(
                url=raw_url,
                preferred_language=pref_lang,
                force_refresh=force_refresh,
            )

            if resp.status == "error":
                status.update(label="Processing Failed", state="error")
                err_msg = resp.error.message if resp.error else "Failed to process video."
                err_code = resp.error.code if resp.error else "ERROR"
                
                if err_code == "YOUTUBE_TRANSCRIPT_BLOCKED":
                    st.error(
                        f"⚠️ YouTube has blocked automated transcript access for this video from the host IP. "
                        f"Details: {err_msg}"
                    )
                elif err_code == "TRANSCRIPT_UNAVAILABLE":
                    st.error(
                        f"⚠️ This video does not have a usable transcript or subtitles available. "
                        f"Details: {err_msg}"
                    )
                else:
                    st.error(f"⚠️ {err_msg}")
            else:
                status.update(label="Video Ready for Questions!", state="complete")
                st.session_state["current_video"] = {
                    "url": raw_url,
                    "video_id": resp.video_id,
                    "title": resp.title or f"YouTube Video ({resp.video_id})",
                    "language": resp.transcript_language or "en",
                    "is_generated": resp.is_generated,
                    "cached": resp.cached,
                }
                # Reset chat history when a new video is loaded
                st.session_state["qa_history"] = []
                st.toast("Video processed successfully!", icon="✅")
                st.rerun()


# ---------------------------------------------------------------------------
# Step 2: Active Video Display & Q&A
# ---------------------------------------------------------------------------
if st.session_state["current_video"]:
    cv = st.session_state["current_video"]
    
    # Display Active Video Card
    st.markdown(
        f"""
        <div class="video-card">
            <div class="video-title">🎬 {cv['title']}</div>
            <div class="video-meta">
                <span><b>Video ID:</b> <code>{cv['video_id']}</code></span>
                <span><b>Transcript Language:</b> <code>{cv['language']}</code></span>
                <span><b>Captions:</b> {'Auto-generated' if cv.get('is_generated') else 'Manual'}</span>
                <span><b>FAISS Index:</b> {'Cached' if cv.get('cached') else 'Built'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Video preview player (collapsible)
    with st.expander("▶️ Watch Video Preview", expanded=False):
        st.video(cv["url"])

    st.markdown("---")
    st.subheader("💬 Ask Questions About This Video")

    # Sample prompt chips
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    with sample_col1:
        if st.button("💡 Summarize the main topics", use_container_width=True):
            st.session_state["auto_query"] = "Summarize the main topics covered in this video in detail."
    with sample_col2:
        if st.button("🔑 What are the key takeaways?", use_container_width=True):
            st.session_state["auto_query"] = "What are the key takeaways from this video?"
    with sample_col3:
        if st.button("❓ What practical advice was given?", use_container_width=True):
            st.session_state["auto_query"] = "What practical advice or recommendations are given in this video?"

    # Question Input
    preset_q = st.session_state.pop("auto_query", "")
    
    q_col, ask_col = st.columns([4, 1])
    with q_col:
        question_input = st.text_input(
            "Ask a question",
            value=preset_q,
            placeholder="e.g., What does the speaker explain about...",
            label_visibility="collapsed",
        )
    with ask_col:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    # Ask Question Action
    if ask_btn or (preset_q and not ask_btn):
        q = (question_input or "").strip()
        if not q:
            st.warning("Please enter a question about the video.")
        elif len(q) > MAX_QUESTION_LENGTH:
            st.error(f"Question is too long (limit is {MAX_QUESTION_LENGTH} characters).")
        else:
            with st.spinner("Retrieving relevant context and generating grounded answer..."):
                resp = rag_service.ask_question(cv["video_id"], q)

                if resp.status == "error":
                    st.error(resp.error.message if resp.error else "Failed to generate answer.")
                else:
                    # Save to session QA history
                    st.session_state["qa_history"].insert(0, {
                        "question": q,
                        "answer": resp.answer,
                        "relevant_context_found": resp.relevant_context_found,
                        "retrieved_sections": [
                            s.model_dump() if hasattr(s, "model_dump") else s.__dict__
                            for s in resp.retrieved_sections
                        ],
                        "status": resp.status,
                    })
                    st.rerun()

    # ---------------------------------------------------------------------------
    # Step 3: Render Q&A History
    # ---------------------------------------------------------------------------
    if st.session_state["qa_history"]:
        st.markdown("### Conversation History")
        
        for idx, item in enumerate(st.session_state["qa_history"]):
            with st.container():
                st.markdown(
                    f"""
                    <div class="qa-box">
                        <div class="question-title">❓ {item['question']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Answer presentation
                st.markdown(item["answer"])
                
                # Sources Section
                if item.get("retrieved_sections"):
                    with st.expander(f"📚 Sources & Timestamps ({len(item['retrieved_sections'])} chunks retrieved)"):
                        st.markdown("**Retrieved Transcript Passages:**")
                        
                        tree_lines = []
                        total_chunks = len(item["retrieved_sections"])
                        for c_idx, sec in enumerate(item["retrieved_sections"]):
                            connector = "└── " if c_idx == total_chunks - 1 else "├── "
                            ts = sec.get("timestamp", "??:??")
                            start_s = sec.get("start_seconds", 0.0)
                            link = get_youtube_timestamp_url(cv["video_id"], start_s)
                            tree_lines.append(f"{connector}[{ts}]({link}) — Chunk #{sec.get('chunk_id')}")

                        st.markdown("\n".join(tree_lines))
                        st.markdown("---")

                        for sec in item["retrieved_sections"]:
                            ts = sec.get("timestamp", "??:??")
                            start_s = sec.get("start_seconds", 0.0)
                            link = get_youtube_timestamp_url(cv["video_id"], start_s)
                            
                            st.markdown(
                                f"**[{ts}]({link})** `Chunk #{sec.get('chunk_id')}` "
                                f"*(starts at {format_seconds(start_s)})*"
                            )
                            st.caption(f'"{sec.get("text")}"')
                            st.write("")
                elif not item.get("relevant_context_found"):
                    st.info("ℹ️ No high-confidence transcript evidence was found for this specific question.")

                st.markdown("---")
else:
    # Empty State Guide
    st.info(
        "👋 **Welcome to AskTube!**\n\n"
        "To get started:\n"
        "1. Paste any YouTube video link with captions above.\n"
        "2. Click **Process Video** to fetch the transcript and index it with FAISS.\n"
        "3. Ask any question to get an AI answer strictly grounded in what was said in the video with timestamped citations."
    )

    # Architecture Overview Callout
    with st.expander("🛠️ How AskTube's RAG Architecture Works", expanded=False):
        st.markdown(
            """
            ```
            YouTube URL
                 ↓
            Video ID Validation & oEmbed Title
                 ↓
            Transcript Retrieval (youtube-transcript-api with language fallback)
                 ↓
            Timestamp-Aware Chunking (1000 chars, 200 overlap)
                 ↓
            Embeddings (intfloat/multilingual-e5-base with passage prefix)
                 ↓
            Vector Index (FAISS Cosine Similarity, disk/RAM cached)
                 ↓
            Semantic Search (Cosine similarity threshold = 0.50, Top-4)
                 ↓
            Prompt Grounding (Strict transcript-only context framing)
                 ↓
            Inference (Qwen/Qwen3-8B via Hugging Face Serverless API)
                 ↓
            Grounded Answer + Clickable Timestamped Citations
            ```
            """
        )
