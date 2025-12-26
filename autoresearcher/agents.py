# autoresearcher/agents.py

from typing import List, Tuple
from .llm_client import LLMClient


class SearcherAgent:
    """
    Given a question and retrieved chunks, pick + summarize the most relevant bits.
    """

    SYSTEM_PROMPT = (
        "You are a research search agent. "
        "You are given a user question and several document chunks. "
        "Your job is to extract only the MOST relevant quotes/snippets from the chunks. "
        "Do not answer the question fully; just provide a set of bullet points with evidence."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str, chunks: List[Tuple[str, float]]) -> str:
        chunks_str = ""
        for i, (text, score) in enumerate(chunks, start=1):
            chunks_str += f"\n[Chunk {i} | score={score:.3f}]\n{text}\n"

        prompt = (
            f"User question:\n{question}\n\n"
            f"Retrieved document chunks:\n{chunks_str}\n\n"
            "Now list the most relevant sentences or paragraphs (with chunk numbers) as bullet points.\n"
        )

        return self.llm.generate(prompt, system=self.SYSTEM_PROMPT)


class CriticAgent:
    """
    Check if the searcher's evidence is enough and identify gaps / missing aspects.
    """

    SYSTEM_PROMPT = (
        "You are a critical research reviewer. "
        "You receive a user question and some evidence bullets from documents. "
        "Your job is to:\n"
        "1) Check if the evidence fully answers the question.\n"
        "2) Identify any missing aspects or uncertainties.\n"
        "3) State clearly if the evidence is likely SUFFICIENT or INSUFFICIENT.\n"
        "Be concise but precise."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str, evidence_bullets: str) -> str:
        prompt = (
            f"User question:\n{question}\n\n"
            f"Evidence bullets from documents:\n{evidence_bullets}\n\n"
            "Now provide your critique and state:\n"
            "- What is well supported.\n"
            "- What is missing.\n"
            "- Final judgment: EVIDENCE SUFFICIENT or EVIDENCE INSUFFICIENT.\n"
        )
        return self.llm.generate(prompt, system=self.SYSTEM_PROMPT)


class WriterAgent:
    """
    Generate final structured answer grounded in the evidence + critic feedback.
    """

    SYSTEM_PROMPT = (
        "You are a research writer. "
        "You must write a clear, well-structured answer grounded ONLY in the provided evidence. "
        "If evidence is missing, clearly say what cannot be concluded.\n\n"
        "Structure your answer with headings and bullet points where appropriate."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str, evidence_bullets: str, critic_review: str) -> str:
        prompt = (
            f"User question:\n{question}\n\n"
            f"Evidence bullets:\n{evidence_bullets}\n\n"
            f"Critic review:\n{critic_review}\n\n"
            "Now write the final answer.\n"
        )
        return self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
