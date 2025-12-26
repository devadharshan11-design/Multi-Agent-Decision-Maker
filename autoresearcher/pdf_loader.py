# autoresearcher/pdf_loader.py

import os
from typing import List, Dict
from pypdf import PdfReader


def list_pdfs(folder: str) -> List[str]:
    """
    Return a list of absolute paths to all .pdf files in the given folder.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    pdfs: List[str] = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                pdfs.append(full)

    return sorted(pdfs)


def load_pdfs(folder: str) -> List[Dict]:
    """
    Load all PDFs in a folder and return a list of page dictionaries:
    {
        "id": str,
        "text": str,
        "metadata": { "source": path, "page": int, "filename": str }
    }
    """
    pdf_paths = list_pdfs(folder)

    if not pdf_paths:
        # This will be surfaced by AutoResearcher as 'Failed to build index: ...'
        raise ValueError(f"No PDFs found in folder: {folder}")

    pages: List[Dict] = []

    for path in pdf_paths:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                # Skip blank pages
                continue

            pages.append(
                {
                    "id": f"{os.path.basename(path)}_p{i+1}",
                    "text": text,
                    "metadata": {
                        "source": path,
                        "page": i + 1,
                        "filename": os.path.basename(path),
                    },
                }
            )

    return pages


def load_pdfs_from_folder(folder: str) -> List[Dict]:
    """
    Thin wrapper used by the orchestrator.
    (Kept separate just so imports match cleanly.)
    """
    return load_pdfs(folder)


def simple_chunk(
    pages: List[Dict],
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Dict]:
    """
    Very simple text chunker.
    Input: list of page dicts from load_pdfs().
    Output: list of chunk dicts:
      {
        "id": str,
        "text": str,
        "metadata": {...}
      }
    """
    chunks: List[Dict] = []
    chunk_counter = 0

    for page in pages:
        text = page["text"]
        meta = page["metadata"]

        start = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size, n)
            chunk_text = text[start:end].strip()
            if not chunk_text:
                break

            chunk_id = f'{page["id"]}_c{chunk_counter}'
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": meta,
                }
            )
            chunk_counter += 1

            if end == n:
                break
            # move forward with overlap
            start = max(0, end - overlap)

    return chunks
