"""
app.py — Explorer Corvit Assistant — Streamlit UI
Layout: permanent left column (unclosable) + main chat area
"""

import os
import uuid
import time
import streamlit as st
from pathlib import Path
from loguru import logger

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explorer Corvit Assistant",
    page_icon="🎓",
    layout="wide",
)

# ─── Local imports ────────────────────────────────────────────────────────────
import config
from agent import CorvitAgent
from rag.ingest import ingest_file, get_collection_stats
from memory.chat_memory import (
    create_session, list_sessions, delete_session,
    get_all_messages, get_session_message_count,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logger.add(
    config.LOGS_DIR / "app.log",
    rotation="10 MB",
    retention="7 days",
    level=config.LOG_LEVEL,
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Remove default padding */
    .block-container { padding-top: 1rem !important; }

    /* Left panel styling */
    .left-panel {
        background: #0f172a;
        border-radius: 12px;
        padding: 1.2rem;
        height: 100vh;
        border-right: 1px solid #1e293b;
        position: sticky;
        top: 0;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #e2e8f0;
        margin: 0;
        font-size: 1.4rem;
    }
    .main-header p { color: #94a3b8; margin: 0; font-size: 0.8rem; }

    /* Chat bubbles */
    .chat-user {
        background: #1e40af;
        color: #fff;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0 0.4rem 10%;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(30,64,175,0.3);
    }
    .chat-assistant {
        background: #1e293b;
        color: #e2e8f0;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.4rem 10% 0.4rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
        border: 1px solid #334155;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Input */
    .stTextInput > div > div > input {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Left panel text colors */
    [data-testid="column"]:first-child * {
        color: #cbd5e1;
    }
    [data-testid="column"]:first-child .stButton > button {
        background: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        font-size: 0.8rem !important;
        text-align: left !important;
    }
    [data-testid="column"]:first-child .stButton > button:hover {
        background: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        create_session(st.session_state.session_id)
    if "agent" not in st.session_state:
        st.session_state.agent = CorvitAgent()
    if "messages" not in st.session_state:
        raw = get_all_messages(st.session_state.session_id)
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in raw
        ]

init_state()


# ─── Layout: Left Column + Main Column ───────────────────────────────────────
left_col, main_col = st.columns([1, 3], gap="small")


# ════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — permanent, cannot be closed
# ════════════════════════════════════════════════════════════════════════════
with left_col:
    st.markdown("""
    <div style="background:#0f172a; padding:1rem; border-radius:12px; border:1px solid #1e293b;">
        <h3 style="color:#e2e8f0; font-family:'Space Grotesk',sans-serif; margin:0; font-size:1.1rem;">
            🎓 Explorer Corvit
        </h3>
        <p style="color:#64748b; font-size:0.75rem; margin:4px 0 0 0;">
            AI Assistant — Corvit Systems Rawalpindi
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # New Chat button
    if st.button("➕  New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        create_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    # Recent chats
    st.markdown("<p style='color:#64748b; font-size:0.78rem; margin:0.5rem 0 0.3rem 0;'>RECENT CHATS</p>", unsafe_allow_html=True)
    sessions = list_sessions()
    for s in sessions[:6]:
        sid    = s["session_id"]
        count  = get_session_message_count(sid)
        active = "🟢 " if sid == st.session_state.session_id else "💬 "
        label  = f"{active}{sid} ({count})"
        if st.button(label, key=f"sess_{sid}", use_container_width=True):
            st.session_state.session_id = sid
            raw = get_all_messages(sid)
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in raw
            ]
            st.rerun()

    st.markdown("<hr style='border-color:#1e293b; margin:0.8rem 0'>", unsafe_allow_html=True)

    # Upload files
    st.markdown("<p style='color:#64748b; font-size:0.78rem; margin:0 0 0.3rem 0;'>📁 UPLOAD CORVIT DATA</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for up_file in uploaded:
            save_path = config.UPLOADS_DIR / up_file.name
            with open(save_path, "wb") as f:
                f.write(up_file.read())
            with st.spinner(f"Processing {up_file.name}..."):
                n = ingest_file(str(save_path))
            st.success(f"✅ {up_file.name}\n{n} chunks added")

    # Knowledge base stats
    stats = get_collection_stats()
    st.markdown(
        f"<p style='color:#475569; font-size:0.72rem; margin:0.3rem 0;'>🧠 {stats['total_chunks']} chunks in knowledge base</p>",
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color:#1e293b; margin:0.8rem 0'>", unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️  Clear Chat", use_container_width=True):
        delete_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages   = []
        create_session(st.session_state.session_id)
        st.rerun()

    st.markdown(
        "<p style='color:#334155; font-size:0.68rem; margin-top:0.5rem;'>Powered by Groq<br>NAVTTC Certified</p>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ════════════════════════════════════════════════════════════════════════════
with main_col:

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Explorer Corvit Assistant</h1>
        <p>Your AI guide for Corvit Systems Rawalpindi — courses, fees, admissions & more</p>
    </div>
    """, unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:2rem 1rem; color:#64748b;">
            <div style="font-size:2.5rem;">🤖</div>
            <h3 style="color:#94a3b8; font-family:'Space Grotesk',sans-serif;">
                Welcome to Explorer Corvit Assistant!
            </h3>
            <p>Ask me anything about Corvit Systems Rawalpindi</p>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; justify-content:center; margin-top:1rem;">
                <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.78rem; border:1px solid #334155;">💰 Fee structure kya hai?</span>
                <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.78rem; border:1px solid #334155;">📅 Admission kab open hoga?</span>
                <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.78rem; border:1px solid #334155;">🖥️ Web dev course?</span>
                <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.78rem; border:1px solid #334155;">🔒 Cybersecurity course?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🎓 {msg["content"]}</div>', unsafe_allow_html=True)

    st.divider()

    # Input area
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "message",
            placeholder="e.g. Python course ki fees kya hai? | What courses are available?",
            label_visibility="collapsed",
            key="user_input",
        )
    with col_send:
        send_clicked = st.button("Send 🚀", use_container_width=True, type="primary")

    # Quick prompts
    quick_prompts = ["Courses?", "Fees?", "NAVTTC?", "Admission?", "Cybersecurity?"]
    qp_cols = st.columns(len(quick_prompts))
    for i, qp in enumerate(quick_prompts):
        if qp_cols[i].button(qp, key=f"qp_{i}", use_container_width=True):
            user_input   = qp
            send_clicked = True

    # Handle send
    if send_clicked and user_input and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("🤔 Thinking..."):
            result = st.session_state.agent.chat(
                session_id=st.session_state.session_id,
                user_query=query,
            )

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        with st.expander("ℹ️ Response details", expanded=False):
            cols = st.columns(4)
            cols[0].metric("Model",    result["model_used"].split("-")[0])
            cols[1].metric("Latency",  f"{result['latency']}s")
            cols[2].metric("Context",  "✅" if result["context_used"]  else "❌")
            cols[3].metric("Fallback", "✅" if result["fallback_used"] else "❌")

        st.rerun()