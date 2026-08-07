import os
import json
import pickle
import streamlit as st
import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHAT_FILE = "chat_history.json"

st.set_page_config(
    page_title="CricRag — AI Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark UI Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Background & Main Layout */
    .stApp {
        background: #090d16;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #0d1322;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(14, 116, 144, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Status Indicator Badge */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #121929 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }

    /* Citation Source Boxes */
    .source-box {
        background: #0b101d;
        border-left: 3px solid #10b981;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 6px;
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    /* Primary Accent Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Chat Input Bar */
    .stChatInputContainer {
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: #0f172a !important;
    }

    /* Hide Default Streamlit Menus */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Save/Load Disk Memory
def load_chat():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat(messages):
    with open(CHAT_FILE, "w") as f:
        json.dump(messages, f, indent=2)

@st.cache_resource(show_spinner="⚡ Initializing Vector Engine...")
def load_rag_assets():
    if not os.path.exists("cricket_index.faiss") or not os.path.exists("cricket_docs.pkl"):
        st.error("Missing index files! Please run `python ingest.py` first.")
        st.stop()
    
    index = faiss.read_index("cricket_index.faiss")
    with open("cricket_docs.pkl", "rb") as f:
        all_docs = pickle.load(f)
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, all_docs, embedder

index, all_docs, embedder = load_rag_assets()

# API Authentication
api_key = os.getenv("GROQ_API_KEY") or st.sidebar.text_input("Enter Groq Key:", type="password")
if not api_key:
    st.info("Add your Groq API Key to `.env` or paste it in the sidebar.")
    st.stop()

groq_client = Groq(api_key=api_key)

# Session State Persistence
if "messages" not in st.session_state:
    st.session_state.messages = load_chat()

# Sidebar Controls & History Overview
st.sidebar.title("🏏 CricRag")
st.sidebar.markdown('<span class="status-pill">🟢 Neural Search Ready</span>', unsafe_allow_html=True)
st.sidebar.markdown("---")

k_retrievals = st.sidebar.slider("Context Depth (Chunks)", min_value=3, max_value=10, value=6)
show_sources = st.sidebar.toggle("Show Citations", value=True)

st.sidebar.markdown("---")

if st.sidebar.button("➕ New Chat Thread"):
    st.session_state.messages = []
    if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)
    st.rerun()

# Sidebar Chat History List
if st.session_state.messages:
    st.sidebar.markdown("**Recent Messages**")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    for q in user_msgs[-5:]:
        st.sidebar.caption(f"💬 {q['content'][:30]}...")

# Top Header Layout
st.markdown("""
<div class="hero-card">
    <h1 class="hero-title">🏏 CricRag Engine</h1>
    <p style="color: #94a3b8; margin: 4px 0 10px 0; font-size: 0.95rem;">Retrieval-Augmented Intelligence for Cricket Records & Analytics</p>
    <div>
        <span class="status-pill">Llama 3.3 70B</span>
        <span class="status-pill">FAISS Vector Store</span>
        <span class="status-pill">Persistent Disk Storage</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Chat Feed
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🏏"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "sources" in msg and show_sources:
            with st.expander("View Referenced Context"):
                for src in msg["sources"]:
                    st.markdown(f"<div class='source-box'><b>{src['title']}</b> ({src['source']}) — Similarity: {src['score']:.3f}</div>", unsafe_allow_html=True)

def retrieve(query, k=6):
    q_emb = embedder.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx != -1:
            doc = all_docs[idx]
            results.append({**doc, "score": float(score)})
    return results

# Handle Query Submissions
if prompt := st.chat_input("Ask any cricket question..."):
    # Display User Message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # RAG Retrieval Process
    with st.spinner("Retrieving facts..."):
        hits = retrieve(prompt, k=k_retrievals)
        context_str = "\n\n---\n\n".join(f"[Source: {h['title']} | {h['source']}]\n{h['text']}" for h in hits)

    # API Payload Assembly
    system_instruction = (
        "You are CricRag, an authoritative cricket knowledge engine. You retain memory of the conversation history.\n"
        "Answer strictly using the CONTEXT provided below and relevant prior messages.\n"
        "State clearly if an answer cannot be found within the provided context."
    )

    api_messages = [{"role": "system", "content": system_instruction}]
    for prev in st.session_state.messages:
        api_messages.append({"role": prev["role"], "content": prev["content"]})

    augmented_user_message = f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {prompt}"
    api_messages.append({"role": "user", "content": augmented_user_message})

    # Display Streaming Assistant Response
    with st.chat_message("assistant", avatar="🏏"):
        message_placeholder = st.empty()
        full_response = ""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.2,
            max_tokens=800,
            stream=True
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

        if show_sources:
            with st.expander("View Referenced Context"):
                for src in hits:
                    st.markdown(f"<div class='source-box'><b>{src['title']}</b> ({src['source']}) — Similarity: {src['score']:.3f}</div>", unsafe_allow_html=True)

    # Save Memory State
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": hits})
    save_chat(st.session_state.messages)