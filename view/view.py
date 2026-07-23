import streamlit as st
import requests
import uuid

# ---- Configuration ----
API_BASE_URL = "http://localhost:8000/api"

# ---- Page Config ----
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    /* Dark theme foundation */
    .stApp {
        background-color: #0d1117;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Chat message containers */
    .user-message {
        background: linear-gradient(135deg, #1a73e8, #1565c0);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.3);
    }

    .assistant-message {
        background-color: #1c2333;
        color: #e6edf3;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid #30363d;
    }

    /* Message row layout */
    .message-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 16px;
        clear: both;
        overflow: hidden;
    }

    .message-row-user {
        justify-content: flex-end;
    }

    .message-row-assistant {
        justify-content: flex-start;
    }

    /* Avatar circles */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
        margin-top: 4px;
    }

    .avatar-user {
        background: linear-gradient(135deg, #1a73e8, #1565c0);
    }

    .avatar-assistant {
        background: linear-gradient(135deg, #238636, #2ea043);
    }

    /* Sidebar branding */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 20px;
    }

    .sidebar-brand h1 {
        color: #58a6ff;
        font-size: 22px;
        margin: 0;
    }

    .sidebar-brand p {
        color: #8b949e;
        font-size: 13px;
        margin: 4px 0 0 0;
    }

    /* New chat button */
    .new-chat-btn button {
        width: 100%;
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-weight: 600 !important;
    }

    /* Session buttons */
    .session-btn button {
        width: 100%;
        text-align: left !important;
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
    }

    .session-btn button:hover {
        background-color: #30363d !important;
    }

    /* Upload section */
    .upload-section {
        border-top: 1px solid #30363d;
        padding-top: 16px;
        margin-top: 16px;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Input styling */
    .stTextInput input, .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Pipeline trace styling */
    .pipeline-trace {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 13px;
        color: #8b949e;
    }

    .pipeline-trace code {
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)


# ---- Session State Initialization ----
def init_session_state():
    """Initialize all session state variables."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}  # {session_id: {"title": str, "messages": list}}

    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""

init_session_state()


# ---- API Helper Functions ----
def api_new_session() -> str:
    """Call POST /api/chat/new to get a fresh session ID."""
    try:
        resp = requests.post(f"{API_BASE_URL}/chat/new", timeout=10)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except Exception:
        return str(uuid.uuid4())


def api_chat(message: str, session_id: str) -> dict:
    """Call POST /api/chat to send a message and get the agent response."""
    resp = requests.post(
        f"{API_BASE_URL}/chat",
        json={"message": message, "session_id": session_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_ingest(file_bytes: bytes, filename: str, source_type: str) -> dict:
    """Call POST /api/ingest to upload and process a document."""
    resp = requests.post(
        f"{API_BASE_URL}/ingest",
        files={"file": (filename, file_bytes, "application/pdf")},
        data={"source_type": source_type},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def api_health() -> dict:
    """Call GET /api/health to check API status."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unreachable"}


# ---- Helper Functions ----
def create_new_session():
    """Create a new chat session and set it as active."""
    session_id = api_new_session()
    st.session_state.sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
    }
    st.session_state.active_session_id = session_id
    return session_id


def get_active_session():
    """Get the active session data or create one if none exists."""
    if not st.session_state.active_session_id:
        create_new_session()
    return st.session_state.sessions[st.session_state.active_session_id]


# ---- Sidebar ----
with st.sidebar:
    # Branding
    st.markdown("""
    <div class="sidebar-brand">
        <h1>🤖 Enterprise RAG</h1>
        <p>Intelligent Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # API Status indicator
    health = api_health()
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.caption(f"{status_color} API Status: {health.get('status', 'unknown')}")

    st.divider()

    # Groq API Key
    groq_key = st.text_input(
        "🔑 Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="Enter your Groq API key...",
    )
    if groq_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = groq_key

    st.divider()

    # New Chat Button
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    # Session History
    st.markdown("##### 💬 Chat Sessions")
    if st.session_state.sessions:
        for sid, sdata in reversed(list(st.session_state.sessions.items())):
            is_active = sid == st.session_state.active_session_id
            label = f"{'▶ ' if is_active else ''}{sdata['title']}"
            if st.button(label, key=f"session_{sid}", use_container_width=True):
                st.session_state.active_session_id = sid
                st.rerun()
    else:
        st.caption("No sessions yet. Click 'New Chat' to start.")

    st.divider()

    # Document Upload Section
    st.markdown("##### 📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload a PDF to ingest",
        type=["pdf"],
        label_visibility="collapsed",
    )
    source_type = st.text_input(
        "Source Type",
        value="general",
        placeholder="e.g., finance, manuals",
    )

    if uploaded_file and st.button("🚀 Ingest Document", use_container_width=True):
        with st.spinner("Ingesting document... This may take a few minutes."):
            try:
                result = api_ingest(uploaded_file.read(), uploaded_file.name, source_type)
                st.success(
                    f"✅ Ingested **{result['filename']}**\n\n"
                    f"Chunks: {result['chunks_created']} | "
                    f"Vectors: {result['vectors_upserted']}"
                )
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")


# ---- Main Chat Area ----
session = get_active_session()

# Chat header
st.markdown("### 🤖 Enterprise RAG Assistant")
st.caption("Ask questions about your ingested documents. The AI will retrieve relevant context and generate answers.")

st.divider()

# Display chat messages using native Streamlit chat components
for msg in session["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])

    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

            # Show retrieved documents dropdown if available
            if msg.get("retrieved_documents"):
                with st.expander(f"📚 View Retrieved Documents ({len(msg['retrieved_documents'])} chunks)"):
                    for j, doc in enumerate(msg["retrieved_documents"]):
                        st.markdown(f"**Chunk {j+1}:**")
                        st.code(doc[:500] + ("..." if len(doc) > 500 else ""), language="markdown")
                        st.divider()

            # Show pipeline trace if available
            if msg.get("plan"):
                with st.expander("⚙️ Pipeline Execution Trace"):
                    st.markdown(
                        f"- **Planner Decision:** `{msg.get('planner_decision', 'N/A')}`\n"
                        f"- **Status:** `{msg.get('status', 'N/A')}`\n"
                        f"- **Plan:** `{' → '.join(msg.get('plan', []))}`\n"
                        f"- **Retrieved Chunks:** `{msg.get('retrieved_chunks', 0)}`"
                    )


# Chat input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add and display user message immediately
    session["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Update session title from first message
    if session["title"] == "New Chat":
        session["title"] = user_input[:40] + ("..." if len(user_input) > 40 else "")

    # Show assistant thinking with spinner inside the chat bubble
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = api_chat(user_input, st.session_state.active_session_id)

                st.markdown(response["answer"])

                # Show retrieved docs dropdown
                docs = response.get("retrieved_documents", [])
                if docs:
                    with st.expander(f"📚 View Retrieved Documents ({len(docs)} chunks)"):
                        for j, doc in enumerate(docs):
                            st.markdown(f"**Chunk {j+1}:**")
                            st.code(doc[:500] + ("..." if len(doc) > 500 else ""), language="markdown")
                            st.divider()

                # Show pipeline trace
                plan = response.get("plan", [])
                if plan:
                    with st.expander("⚙️ Pipeline Execution Trace"):
                        st.markdown(
                            f"- **Planner Decision:** `{response.get('planner_decision', 'N/A')}`\n"
                            f"- **Status:** `{response.get('status', 'N/A')}`\n"
                            f"- **Plan:** `{' → '.join(plan)}`\n"
                            f"- **Retrieved Chunks:** `{response.get('retrieved_chunks', 0)}`"
                        )

                # Save to session state
                session["messages"].append({
                    "role": "assistant",
                    "content": response["answer"],
                    "planner_decision": response.get("planner_decision", ""),
                    "status": response.get("status", ""),
                    "plan": response.get("plan", []),
                    "retrieved_chunks": response.get("retrieved_chunks", 0),
                    "retrieved_documents": response.get("retrieved_documents", []),
                })

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to the API server. Make sure it's running with: `uvicorn main:app --reload --port 8000`")
                session["messages"].append({
                    "role": "assistant",
                    "content": "⚠️ Cannot connect to the API server.",
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                session["messages"].append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
                })

