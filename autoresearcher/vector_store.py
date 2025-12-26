import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = []

    def build(self, chunks):
        print("[VectorStore] Building embeddings...")

        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.texts = chunks
        print(f"[VectorStore] Index built with {len(chunks)} chunks.")

    def search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Index not built.")

        q_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_vector, top_k)

        return [self.texts[i] for i in indices[0]]
