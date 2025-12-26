import os
from .pdf_loader import load_pdfs_from_folder, simple_chunk
from .vector_store import VectorStore
from .llm_client import LLMClient


class AutoResearcher:
    def __init__(self):
        self.indexes = {}
        self.vectorstores = {}
        self.llm = LLMClient()

    def index_exists(self, name):
        return name in self.indexes

    def build_index(self, name, folder):
        print(f"[AutoResearcher] Loading PDFs from {folder}")
        pages = load_pdfs_from_folder(folder)

        if not pages:
            raise ValueError("No PDF pages extracted.")

        print("[AutoResearcher] Chunking text...")
        chunks = simple_chunk(pages)

        if not chunks:
            raise ValueError("No text chunks created.")

        print("[AutoResearcher] Building vector store...")
        store = VectorStore()
        store.build(chunks)

        self.indexes[name] = True
        self.vectorstores[name] = store

    def answer(self, query, index_name):
        if index_name not in self.vectorstores:
            raise ValueError("Index not found.")

        print("[AutoResearcher] Retrieving chunks...")
        store = self.vectorstores[index_name]
        retrieved = self.vector_store.search(question)

# SAFE JOIN (fix dict/string mismatch bug)
        context = "\n\n".join(
        chunk["text"] if isinstance(chunk, dict) else str(chunk)
        for chunk in retrieved
)

        clean = [chunk["text"] if isinstance(chunk, dict) else str(chunk) for chunk in retrieved]
        context = "\n\n".join(clean)


        prompt = f"""
You are a research assistant.

Use the context below to answer the question.

--- CONTEXT ---
{context}

--- QUESTION ---
{query}

--- ANSWER ---
"""

        return self.llm.generate(prompt)
