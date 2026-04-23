# 📚 RAG Study Assistant

A **Retrieval-Augmented Generation (RAG)** application that turns your PDF courses and notes into an interactive AI study assistant. Ask questions in natural language and get precise, sourced answers — all running locally.

## 🎯 Features

- 📄 Upload any PDF course or document
- 🔍 Semantic search over your documents using FAISS vector store
- 🤖 Natural language Q&A powered by LLM (OpenAI or local Ollama)
- 📊 Source citations with page references
- 💬 Conversation memory across a session
- 🌐 Clean Flask web interface

## 🛠️ Stack

| Layer | Technology |
|---|---|
| Embedding | `sentence-transformers` (MiniLM-L6) |
| Vector store | FAISS |
| LLM orchestration | LangChain |
| LLM backend | OpenAI GPT-4o / Ollama (local) |
| PDF parsing | PyMuPDF |
| Web server | Flask |

## 🚀 Quick Start

```bash
git clone https://github.com/RisKyAyO/rag-study-assistant
cd rag-study-assistant
pip install -r requirements.txt
cp .env.example .env   # Add your OpenAI key (or configure Ollama)
python app.py
```

Open `http://localhost:5000`, upload your PDF, and start asking questions.

## 📐 Architecture

```
PDF Input
   ↓
PyMuPDF (text extraction)
   ↓
Text Chunking (LangChain RecursiveCharacterTextSplitter)
   ↓
Embeddings (sentence-transformers)
   ↓
FAISS Vector Store ──→ Similarity Search (top-k chunks)
                                ↓
              LLM (GPT-4o / Ollama) + Retrieved Context
                                ↓
                        Answer + Sources
```

## 📸 Demo

> Upload `cours_algo.pdf` → Ask *"Explain quicksort complexity"* → Get a precise answer with page references.

## 📁 Project Structure

```
rag-study-assistant/
├── app.py              # Flask entry point
├── rag_engine.py       # Core RAG pipeline
├── pdf_loader.py       # PDF parsing & chunking
├── requirements.txt
├── .env.example
└── templates/
    └── index.html      # Web UI
```

## 🔮 Roadmap

- [ ] Multi-document querying
- [ ] Export Q&A history as study flashcards
- [ ] Speech-to-text input (Web Speech API)
- [ ] Support for DOCX/Markdown files

## 📝 License

MIT
