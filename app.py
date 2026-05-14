import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# 1. LOAD ENV
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY_ENV = os.getenv("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
# 2. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="NoteBot", page_icon="📚", layout="wide")

st.markdown("""
<style>
  .source-box {
    background: #1e1e2e;
    border-left: 4px solid #7c3aed;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #cdd6f4;
    margin-top: 6px;
  }
  .rate-warning {
    color: #f38ba8;
    font-weight: bold;
  }
  .tag {
    background: #313244;
    color: #cba6f7;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 0.75rem;
    margin-right: 4px;
  }
</style>
""", unsafe_allow_html=True)

st.header("NoteBot - Your AI Study Buddy")

# ─────────────────────────────────────────────
# 3. SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "vector_store_map": {},   # filename -> FAISS store
    "active_files": [],       # list of currently selected filenames
    "question_count": 0,      # for rate limiting
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# 4. CONSTANTS
# ─────────────────────────────────────────────
MAX_QUESTIONS_PER_SESSION = 20
FAISS_INDEX_DIR = "faiss_indexes"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("My Notes")

    # --- BYOK: User provides their own API key ---
    st.subheader("API Key")
    user_api_key = st.text_input(
        "Enter your Groq API key (optional)",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com. If left blank, a shared demo key is used (limited)."
    )
    GROQ_API_KEY = user_api_key.strip() if user_api_key.strip() else GROQ_API_KEY_ENV

    st.caption("Using your own key = no usage limits for you.")

    st.divider()

    # --- Multi-PDF Upload ---
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()
    st.subheader("Settings")

    model_choice = st.selectbox(
        "Choose Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b" ,],
        index=1
    )
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.0, 0.1)

    st.divider()
    st.metric(
        "Questions Used",
        f"{st.session_state.question_count} / {MAX_QUESTIONS_PER_SESSION}"
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.question_count = 0
        st.success("Chat cleared!")

# ─────────────────────────────────────────────
# 6. EMBEDDINGS (cached so it loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ─────────────────────────────────────────────
# 7. PDF PROCESSING
# ─────────────────────────────────────────────
def get_faiss_path(filename: str) -> str:
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    return os.path.join(FAISS_INDEX_DIR, safe_name)

def process_pdf(uploaded_file) -> FAISS | None:
    """Returns a FAISS vector store for the given PDF. Uses disk cache if available."""
    try:
        faiss_path = get_faiss_path(uploaded_file.name)

        if os.path.exists(faiss_path):
            store = FAISS.load_local(
                faiss_path, embeddings, allow_dangerous_deserialization=True
            )
            return store

        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if not text.strip():
            st.sidebar.warning(
                f"⚠️ '{uploaded_file.name}' has no extractable text "
                "(scanned PDF?). Skipping."
            )
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)

        store = FAISS.from_texts(chunks, embeddings)
        store.save_local(faiss_path)
        return store

    except Exception as e:
        st.sidebar.error(f"Error processing '{uploaded_file.name}': {e}")
        return None

# ─────────────────────────────────────────────
# 8. PROCESS UPLOADED FILES
# ─────────────────────────────────────────────
if uploaded_files:
    newly_added = False
    for f in uploaded_files:
        if f.name not in st.session_state.vector_store_map:
            with st.spinner(f"Indexing {f.name}..."):
                store = process_pdf(f)
                if store:
                    st.session_state.vector_store_map[f.name] = store
                    newly_added = True

    # Let user pick which PDFs to query
    available = list(st.session_state.vector_store_map.keys())
    with st.sidebar:
        st.divider()
        st.subheader("📑 Active Documents")
        selected = st.multiselect(
            "Select PDFs to query",
            options=available,
            default=available
        )
        st.session_state.active_files = selected

    if newly_added:
        st.session_state.chat_history = []
        st.session_state.question_count = 0

# ─────────────────────────────────────────────
# 9. DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-box">'
                        f'<span class="tag">Chunk {i} · {src["file"]}</span><br><br>'
                        f'{src["text"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ─────────────────────────────────────────────
# 10. CHAT SECTION
# ─────────────────────────────────────────────
no_stores = len(st.session_state.vector_store_map) == 0
no_active = len(st.session_state.active_files) == 0

if no_stores:
    st.info("Upload a PDF from the sidebar to start chatting!")
elif no_active:
    st.warning("Select at least one PDF to query from the sidebar.")
else:
    # Rate limit check
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.markdown(
            '<p class="rate-warning">Session limit reached (20 questions). '
            'Clear chat history to continue, or use your own API key for unlimited access.</p>',
            unsafe_allow_html=True
        )
    else:
        user_query = st.chat_input("Ask something about your notes...")

        if user_query:
            # Input validation
            if len(user_query.strip()) < 3:
                st.warning("Please enter a valid question.")
                st.stop()

            if len(user_query) > 500:
                st.warning("Question too long (max 500 characters).")
                st.stop()

            # Show user message
            with st.chat_message("user"):
                st.write(user_query)
            st.session_state.chat_history.append({
                "role": "user", "content": user_query, "sources": []
            })
            st.session_state.question_count += 1

            # ── Semantic search across ALL active PDFs ──
            all_chunks = []
            for fname in st.session_state.active_files:
                store = st.session_state.vector_store_map.get(fname)
                if store:
                    results = store.similarity_search(user_query, k=3)
                    for doc in results:
                        all_chunks.append({
                            "file": fname,
                            "text": doc.page_content
                        })

            # Build context string
            context_text = "\n\n".join(
                [f"[From: {c['file']}]\n{c['text']}" for c in all_chunks]
            )

            # Build history string (last 6 turns)
            history_text = ""
            for h in st.session_state.chat_history[-6:]:
                role = "Student" if h["role"] == "user" else "Tutor"
                history_text += f"{role}: {h['content']}\n"

            # ── LLM ──
            try:
                if not GROQ_API_KEY:
                    st.error("No API key found. Please enter your Groq API key in the sidebar.")
                    st.stop()

                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=model_choice,
                    temperature=temperature
                )

                prompt = ChatPromptTemplate.from_template("""
You are a helpful AI tutor. Answer ONLY using the provided context.
If the answer is not found, say: "I don't find that in your notes."

Previous conversation:
{history}

Context from notes:
{context}

Question: {input}
""")
                chain = prompt | llm | StrOutputParser()

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        output = chain.invoke({
                            "input": user_query,
                            "context": context_text,
                            "history": history_text
                        })
                    st.write(output)

                    # ── Source Citations ──
                    if all_chunks:
                        with st.expander("📄 View Sources"):
                            for i, src in enumerate(all_chunks[:4], 1):
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<span class="tag">Chunk {i} · {src["file"]}</span>'
                                    f'<br><br>{src["text"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": output,
                    "sources": all_chunks[:4]
                })

            except Exception as e:
                st.error(f"LLM Error: {e}. Please check your API key or try again.")