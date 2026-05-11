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
# 1. LOAD ENV VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────────
# 2. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NoteBot",
    page_icon="📚",
    layout="wide"
)

st.header("📚 NoteBot - Your AI Study Buddy")

# ─────────────────────────────────────────────
# 3. SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ─────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.title("📂 My Notes")

    file = st.file_uploader(
        "Upload notes PDF",
        type="pdf"
    )

    st.divider()

    st.subheader("⚙️ Settings")

    model_choice = st.selectbox(
        "Choose Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it"

        ],
        index=1
    )

    temperature = st.slider(
        "Creativity (Temperature)",
        0.0,
        1.0,
        0.0,
        0.1
    )

    st.divider()

    if st.button("🗑️ Clear Chat History"):

        st.session_state.chat_history = []

        st.success("Chat cleared!")

# ─────────────────────────────────────────────
# 5. FAISS STORAGE
# ─────────────────────────────────────────────
FAISS_INDEX_DIR = "faiss_indexes"

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 6. EMBEDDING MODEL
# ─────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ─────────────────────────────────────────────
# 7. GET FAISS PATH
# ─────────────────────────────────────────────
def get_faiss_path(filename: str):

    safe_name = (
        filename
        .replace(" ", "_")
        .replace(".pdf", "")
    )

    return os.path.join(
        FAISS_INDEX_DIR,
        safe_name
    )

# ─────────────────────────────────────────────
# 8. PROCESS PDF
# ─────────────────────────────────────────────
def process_pdf(uploaded_file):

    faiss_path = get_faiss_path(uploaded_file.name)

    # Load existing index
    if os.path.exists(faiss_path):

        vector_store = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        st.sidebar.success(
            f"Loaded saved index for: {uploaded_file.name}"
        )

        return vector_store

    # Create new index
    with st.spinner("Reading and indexing your PDF..."):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        vector_store = FAISS.from_texts(
            chunks,
            embeddings
        )

        vector_store.save_local(faiss_path)

        st.sidebar.success(
            f" Indexed & saved: {uploaded_file.name}"
        )

    return vector_store

# ─────────────────────────────────────────────
# 9. PROCESS NEW FILE
# ─────────────────────────────────────────────
if file is not None:

    if st.session_state.current_file != file.name:

        st.session_state.vector_store = process_pdf(file)

        st.session_state.current_file = file.name

        st.session_state.chat_history = []

# ─────────────────────────────────────────────
# 10. DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

# ─────────────────────────────────────────────
# 11. CHAT SECTION
# ─────────────────────────────────────────────
if st.session_state.vector_store is None:

    st.info(" Upload a PDF from sidebar to start chatting!")

else:

    user_query = st.chat_input(
        "Ask something about your notes..."
    )

    if user_query:

        # Show user message
        with st.chat_message("user"):

            st.write(user_query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        # Similarity search
        matching_chunks = (
            st.session_state.vector_store
            .similarity_search(user_query, k=4)
        )

        # History text
        history_text = ""

        for h in st.session_state.chat_history[-6:]:

            role = (
                "Student"
                if h["role"] == "user"
                else "Tutor"
            )

            history_text += (
                f"{role}: {h['content']}\n"
            )

        # Convert docs into plain text
        context_text = "\n\n".join(
            [doc.page_content for doc in matching_chunks]
        )

        # LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_choice,
            temperature=temperature
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI tutor.

Answer ONLY using the provided context.

If answer is not found, say:
"I don't find that in your notes."

Previous conversation:
{history}

Context:
{context}

Question:
{input}
"""
        )

        # Chain
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                output = chain.invoke({
                    "input": user_query,
                    "context": context_text,
                    "history": history_text
                })

            st.write(output)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": output
        })