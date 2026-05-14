# 📚 NoteBot — AI-Powered Study Assistant

NoteBot is a RAG (Retrieval-Augmented Generation) chatbot that lets you upload your study PDFs and ask questions about them in natural language. Built with LangChain, Groq LLMs, HuggingFace Embeddings, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)

## 🚀 Live Demo
👉 [Click here to try NoteBot](https://your-app-link.streamlit.app) ← replace after deploying

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-PDF Support | Upload and query across multiple PDFs simultaneously |
| 🔍 Source Citations | Every answer shows which chunk/document it came from |
| 💬 Chat History | Context-aware multi-turn conversations |
| 💾 FAISS Persistence | PDFs are indexed once and cached to disk |
| 🔑 BYOK | Users can enter their own Groq API key |
| 🛡️ Rate Limiting | 20 questions/session to prevent API abuse |
| ⚠️ Error Handling | Graceful handling of bad PDFs, missing keys, API failures |
| 🧠 Free Stack | Uses Groq (free LLM) + HuggingFace embeddings (no OpenAI cost) |

---

## 🏗️ Architecture

```
PDF Upload
    ↓
PyPDF2 (Text Extraction)
    ↓
RecursiveCharacterTextSplitter (Chunking)
    ↓
HuggingFace all-MiniLM-L6-v2 (Embeddings)
    ↓
FAISS Vector Store (Saved to Disk)
    ↓
User Query → Similarity Search → Top 3 Chunks per PDF
    ↓
Groq LLM (llama / gemma) + LangChain Prompt
    ↓
Answer + Source Citations → Streamlit Chat UI
```

---

## 🛠️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/notebot.git
cd notebot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env` file
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run notebot.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Under **Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy** — you get a live public URL

---

## 📁 Project Structure

```
notebot/
├── notebot.py          # Main application
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this!)
├── .gitignore          # Ignores .env and faiss_indexes/
├── faiss_indexes/      # Auto-created, stores vector indexes
└── README.md
```

---

## 🔒 .gitignore (important!)

Make sure your `.gitignore` contains:
```
.env
faiss_indexes/
__pycache__/
*.pyc
venv/
```

---

## 🧰 Tech Stack

- **Frontend** — Streamlit
- **LLM** — Groq (Llama 3, Gemma 2) — free tier
- **Embeddings** — HuggingFace `all-MiniLM-L6-v2` — free, runs locally
- **Vector DB** — FAISS (local)
- **Framework** — LangChain

---

## 👤 Author

Made by [Your Name](https://github.com/adityaj2006)