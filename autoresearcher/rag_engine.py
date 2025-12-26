import os
import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, model_name="llama3.1"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = model_name
        self.stores = {}

    def list_pdfs(self, folder):
        return [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".pdf")]

    def load_pdfs(self, folder):
        texts = []
        for pdf in self.list_pdfs(folder):
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        return texts

    def chunk(self, texts, size=500):
        chunks = []
        for text in texts:
            words = text.split()
            for i in range(0, len(words), size):
                chunks.append(" ".join(words[i:i+size]))
        return chunks

    def embed(self, chunks):
        return self.embedder.encode(chunks)

    def build_index(self, name, folder):
        pages = self.load_pdfs(folder)
        chunks = self.chunk(pages)
        embeddings = self.embed(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        self.stores[name] = (index, chunks)
        return len(chunks)

    def query_ollama(self, context, question):
        payload = {
            "model": self.model,
            "prompt": f"Document context:\n{context}\n\nQuestion:\n{question}",
            "stream": False
        }

        r = requests.post("http://localhost:11434/api/generate", json=payload)
        return r.json()["response"]

    def retrieve(self, name, query, k=4):
        index, chunks = self.stores[name]
        vec = self.embed([query])
        _, idx = index.search(vec, k)
        return "\n".join([chunks[i] for i in idx[0]])

    def answer(self, name, question):
        context = self.retrieve(name, question)
        return self.query_ollama(context, question)
