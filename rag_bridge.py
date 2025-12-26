# rag_bridge.py

from autoresearcher.orchestrator import run_pipeline


class RAGBridge:
    def __init__(self):
        # later you can store config here if needed
        pass

    def query(self, question: str, files=None):
        """
        Bridge between the multi-agent system and the AutoResearcher RAG pipeline.

        - If `files` is None: AutoResearcher will just do web / search based answering.
        - If `files` is a list of file paths: it will also use those docs (PDFs, etc.).
        """
        return run_pipeline(question, files=files)
