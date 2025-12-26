# rag_engine.py
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from llm_client import call_llm  # local LLaMA via Ollama


def load_pdf_text(path: str) -> str:
    """Read a single PDF into plain text."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Very simple text chunker: split by words into overlapping chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks


class RAGEngine:
    """
    Minimal in-memory RAG engine.

    - Uses SentenceTransformer to embed chunks
    - Uses FAISS for similarity search
    - Uses local LLaMA (via call_llm) to answer based on retrieved chunks
    """

    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2"):
        # ✅ THIS is what your UI expects:
        self.indexes: Dict[str, Dict] = {}  # name -> {"chunks": [...], "index": faiss_idx, "emb": np.ndarray}
        self.encoder = SentenceTransformer(encoder_model)

    # ---------- INDEXING ----------

    def list_pdfs(self, folder: str) -> List[str]:
        """Return list of .pdf files in folder."""
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        ]

    def build_index(self, name: str, folder: str) -> int:
        """
        Build a vector index from all PDFs in `folder`.
        Returns number of chunks.
        """
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")

        pdfs = self.list_pdfs(folder)
        if not pdfs:
            raise ValueError(f"No PDFs found in folder: {folder}")

        all_chunks: List[str] = []

        for path in pdfs:
            text = load_pdf_text(path)
            chunks = simple_chunk(text)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No text chunks extracted from PDFs.")

        # Embed chunks
        print("[RAG] Encoding chunks with SentenceTransformer...")
        embeddings = self.encoder.encode(all_chunks, show_progress_bar=True)
        embeddings = np.asarray(embeddings).astype("float32")

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Store in memory
        self.indexes[name] = {
            "chunks": all_chunks,
            "index": index,
            "embeddings": embeddings,
        }

        print(f"[RAG] Indexed {len(all_chunks)} chunks under name '{name}'")
        return len(all_chunks)

    # ---------- QUERYING ----------

    def has_index(self, name: str) -> bool:
        return name in self.indexes

    def query(self, name: str, question: str, top_k: int = 5) -> Tuple[str, List[str]]:
        """
        Retrieve top_k relevant chunks for the question.
        Returns (joined_context, list_of_chunks).
        """
        if name not in self.indexes:
            raise ValueError(f"Index '{name}' not found. Build it first.")

        data = self.indexes[name]
        index = data["index"]
        chunks = data["chunks"]

        q_emb = self.encoder.encode([question]).astype("float32")
        D, I = index.search(q_emb, top_k)

        selected = []
        for idx in I[0]:
            if 0 <= idx < len(chunks):
                selected.append(chunks[idx])

        context = "\n\n---\n\n".join(selected)
        return context, selected

    def answer(self, name: str, question: str, top_k: int = 5) -> str:
        """
        Full RAG: retrieve context, then ask local LLaMA to answer.
        """
        context, _ = self.query(name, question, top_k=top_k)

        prompt = f"""You are a precise research assistant.

You will answer the user's question *only* using the context from the PDF chunks below.
If something is not supported by the context, say you don't know.

[QUESTION]
{question}

[CONTEXT]
{context}

Now provide a clear, detailed, well-structured answer grounded in this context.
"""
        return call_llm(prompt)
