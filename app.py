"""
app.py — Explorer Corvit Assistant — Streamlit UI
ChatGPT-style interface with file upload, session memory, and RAG
"""

import os
import uuid
import time
import streamlit as st
from pathlib import Path
from loguru import logger

# ─── Page Config (must be FIRST Streamlit call) ──────────────────────────────
# "expanded" is respected ONLY if we also clear the browser localStorage key.
# The proper fix: use a .streamlit/config.toml to permanently set defaultWideSidebar.
st.set_page_config(
    page_title="Explorer Corvit Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SIDEBAR ALWAYS OPEN FIX ─────────────────────────────────────────────────
# Streamlit saves sidebar state to browser localStorage key "sidebarState".
# We reset it to "open" before the page finishes rendering.
st.markdown("""
<script>
(function() {
    try {
        // Clear Streamlit's stored sidebar preference so "expanded" always wins
        Object.keys(localStorage).forEach(function(key) {
            if (key.toLowerCase().includes('sidebar')) {
                localStorage.removeItem(key);
            }
        });
    } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)

# ─── Local imports ────────────────────────────────────────────────────────────
import config
from agent          import CorvitAgent
from rag.ingest     import ingest_file, get_collection_stats
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

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #e2e8f0;
        margin: 0;
        font-size: 1.6rem;
    }
    .main-header p { color: #94a3b8; margin: 0; font-size: 0.85rem; }

    .chat-user {
        background: #1e40af;
        color: #fff;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.4rem 0 0.4rem 15%;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(30,64,175,0.3);
    }
    .chat-assistant {
        background: #1e293b;
        color: #e2e8f0;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.4rem 15% 0.4rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
        border: 1px solid #334155;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    [data-testid="stSidebar"] { background: #0f172a !important; }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    .stTextInput > div > div > input {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    .stButton > button { border-radius: 8px !important; font-weight: 500 !important; }
    /* Make sidebar toggle arrow always visible and bigger */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 30px !important;
    height: 60px !important;
    background: #3b82f6 !important;
    border-radius: 0 8px 8px 0 !important;
    top: 50% !important;
}
[data-testid="collapsedControl"] svg {
    fill: white !important;
    width: 20px !important;
    height: 20px !important;
}
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
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


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Explorer Corvit")
    st.caption("AI Assistant for Corvit Systems Rawalpindi")
    st.divider()

    if st.button("➕  New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        create_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Recent Chats**")
    sessions = list_sessions()
    for s in sessions[:8]:
        sid   = s["session_id"]
        count = get_session_message_count(sid)
        label = f"💬 {sid}  ({count} msgs)"
        active = " ← current" if sid == st.session_state.session_id else ""
        if st.button(f"{label}{active}", key=f"sess_{sid}", use_container_width=True):
            st.session_state.session_id = sid
            raw = get_all_messages(sid)
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in raw
            ]
            st.rerun()

    st.divider()

    st.markdown("**📁 Upload Corvit Data**")
    uploaded = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for up_file in uploaded:
            save_path = config.UPLOADS_DIR / up_file.name
            with open(save_path, "wb") as f:
                f.write(up_file.read())
            with st.spinner(f"Ingesting {up_file.name}..."):
                n = ingest_file(str(save_path))
            st.success(f"✅ {up_file.name} — {n} chunks added")

    stats = get_collection_stats()
    st.caption(f"🧠 Knowledge base: **{stats['total_chunks']}** chunks")
    st.divider()

    if st.button("🗑️  Clear Current Chat", use_container_width=True):
        delete_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages   = []
        create_session(st.session_state.session_id)
        st.rerun()

    st.divider()
    st.caption("Powered by Groq • NAVTTC Certified Courses")


# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <h1>🎓 Explorer Corvit Assistant</h1>
        <p>Your AI guide for Corvit Systems Rawalpindi — courses, fees, admissions & more</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #64748b;">
        <div style="font-size: 3rem;">🤖</div>
        <h3 style="color:#94a3b8; font-family:'Space Grotesk',sans-serif;">
            Welcome to Explorer Corvit Assistant!
        </h3>
        <p>Ask me anything about Corvit Systems Rawalpindi</p>
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; justify-content:center; margin-top:1.5rem;">
            <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.8rem; border:1px solid #334155;">💰 Fee structure kya hai?</span>
            <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.8rem; border:1px solid #334155;">📅 Admission kab open hoga?</span>
            <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.8rem; border:1px solid #334155;">🖥️ Web dev course details?</span>
            <span style="background:#1e293b; color:#94a3b8; padding:0.4rem 1rem; border-radius:99px; font-size:0.8rem; border:1px solid #334155;">🔒 Cybersecurity course hai?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">🎓 {msg["content"]}</div>', unsafe_allow_html=True)

st.divider()
col_input, col_send = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Ask anything about Corvit...",
        placeholder="e.g. Python course ki fees kya hai? | What courses are available?",
        label_visibility="collapsed",
        key="user_input",
    )
with col_send:
    send_clicked = st.button("Send 🚀", use_container_width=True, type="primary")

quick_prompts = ["Courses available?", "Fee structure?", "NAVTTC certification?", "Admission process?", "Cybersecurity course?"]
qp_cols = st.columns(len(quick_prompts))
for i, qp in enumerate(quick_prompts):
    if qp_cols[i].button(qp, key=f"qp_{i}", use_container_width=True):
        user_input   = qp
        send_clicked = True

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