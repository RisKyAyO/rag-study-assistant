"""
RAG Engine - Core pipeline for Retrieval-Augmented Generation.
"""

import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from pdf_loader import extract_text_from_pdf


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5


class RAGEngine:
    """End-to-end RAG pipeline: ingest -> embed -> retrieve -> generate."""

    def __init__(self, use_local_llm: bool = False):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", return_messages=True, output_key="answer"
        )
        if use_local_llm:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="mistral")
        else:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )

    def ingest_pdf(self, pdf_path: str) -> int:
        """Parse a PDF, chunk it, embed, and store in FAISS. Returns chunk count."""
        raw_pages = extract_text_from_pdf(pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.create_documents(
            texts=[p["text"] for p in raw_pages],
            metadatas=[{"page": p["page"], "source": pdf_path} for p in raw_pages],
        )
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
        self._build_chain()
        return len(docs)

    def _build_chain(self):
        retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": 20}
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and return answer + source references."""
        if self.chain is None:
            raise RuntimeError("No document ingested yet. Call ingest_pdf() first.")
        result = self.chain.invoke({"question": question})
        sources = [
            {"page": doc.metadata.get("page", "?"), "excerpt": doc.page_content[:120]}
            for doc in result.get("source_documents", [])
        ]
        return {"answer": result["answer"], "sources": sources}

    def reset_memory(self):
        self.memory.clear()
