# ui_app.py
import os
import time
import re
from typing import Optional, Tuple, List

import streamlit as st

from rag_engine import RAGEngine          # uses local LLaMA via Ollama
from llm_client import call_gemini       # your Gemini client


# ---------- Helper: parse SCORE: x.y from evaluator output ----------

def extract_score(text: str) -> float:
    """
    Extracts SCORE: x.y from a block of text.
    Returns -1.0 if not found.
    """
    match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return -1.0


# ---------- Helper: evaluation using Gemini -------------------------

def evaluate_answer(
    question: str,
    final_answer: str,
    used_rag: bool,
    mode: str,
) -> Tuple[str, float]:
    """
    Ask Gemini to evaluate how good the final answer is.
    Returns (full_evaluation_text, numeric_score_0_to_10).
    """
    system_prompt = (
        "You are an expert evaluator for AI assistant responses.\n"
        "You will receive:\n"
        "- The mode (engineering / policy / research)\n"
        "- Whether document RAG was used (local LLaMA over PDFs)\n"
        "- The user's question\n"
        "- The assistant's final answer\n\n"
        "Your job:\n"
        "1) Briefly comment on correctness, depth, clarity, and alignment with the requested mode.\n"
        "2) If RAG was used, judge whether the answer seems grounded in a PDF summary.\n"
        "3) List clear strengths and weaknesses.\n"
        "4) At the very end, output a single line:\n"
        "   SCORE: x.y\n"
        "Where x.y is a score from 0.0 (very poor) to 10.0 (excellent).\n"
        "Do NOT omit the SCORE line."
    )

    user_content = (
        f"MODE: {mode}\n"
        f"USED_RAG: {used_rag}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"FINAL ANSWER:\n{final_answer}\n"
    )

    eval_text = call_gemini(system_prompt, user_content)
    score = extract_score(eval_text)
    return eval_text, score


# ---------- Initialize shared objects in session_state --------------

def get_rag_engine() -> RAGEngine:
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    return st.session_state.rag_engine


def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts with interaction data


# ---------- Main Streamlit app -------------------------------------

def main():
    st.set_page_config(
        page_title="Hybrid Multi-Agent + RAG Assistant",
        layout="wide",
    )

    st.title("🧠 Hybrid Multi-Agent Assistant (Gemini + Local LLaMA RAG)")
    st.caption(
        "Gemini handles decision-making / reasoning. "
        "Local LLaMA (via Ollama) handles PDF-grounded summarization and Q&A."
    )

    init_history()
    rag = get_rag_engine()

    # ---- Sidebar: configuration & PDF index management ----
    with st.sidebar:
        st.header("⚙️ Configuration")

        mode = st.selectbox(
            "Select mode",
            ["engineering", "policy", "research"],
            index=2,  # default research
        )

        use_rag = st.checkbox("Use PDF RAG (local LLaMA)", value=True)

        st.markdown("---")
        st.subheader("📄 PDF Index")

        index_name = st.text_input(
            "Index name",
            value="control_papers",
            help="Logical name for this collection of PDFs.",
        )

        st.markdown("**Build / update index from PDFs**")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        build_index_clicked = st.button("📚 Build / Rebuild Index")

        if build_index_clicked:
            if not index_name.strip():
                st.error("Please provide a non-empty index name.")
            elif not uploaded_files:
                st.error("Please upload at least one PDF.")
            else:
                # Save uploaded PDFs to a local folder per index
                base_dir = "uploaded_pdfs"
                os.makedirs(base_dir, exist_ok=True)
                index_dir = os.path.join(base_dir, index_name)
                os.makedirs(index_dir, exist_ok=True)

                # Clear old files
                for fname in os.listdir(index_dir):
                    try:
                        os.remove(os.path.join(index_dir, fname))
                    except Exception:
                        pass

                for up in uploaded_files:
                    save_path = os.path.join(index_dir, up.name)
                    with open(save_path, "wb") as f:
                        f.write(up.getbuffer())

                st.info(f"Saved {len(uploaded_files)} PDF(s) to {index_dir}")
                st.write("[RAG] Building index with local LLaMA embeddings...")
                try:
                    count = rag.build_index(index_name, index_dir)
                    st.success(f"Index '{index_name}' ready with {count} chunks.")
                except Exception as e:
                    st.error(f"Failed to build index: {e}")

        st.markdown("---")
        st.subheader("📊 Evaluation controls")
        auto_evaluate = st.checkbox(
            "Auto-evaluate every answer (Gemini-based SCORE: 0–10)",
            value=True,
        )

    # ---- Main area: chat-like interaction ----
    st.subheader("💬 Ask a question")

    question = st.text_area(
        "Your question / task",
        value="Summarise this PDF in detail",
        height=100,
    )

    col_run, col_clear = st.columns([1, 1])

    with col_run:
        run_btn = st.button("🚀 Run Hybrid System", type="primary")

    with col_clear:
        if st.button("🧹 Clear history"):
            st.session_state.history = []
            st.success("History cleared.")

    if run_btn:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            # --- Run pipeline ---
            used_rag_flag = False
            rag_answer: Optional[str] = None
            rag_time = 0.0
            gemini_time = 0.0
            eval_time = 0.0

            # 1) Optional: PDF RAG via local LLaMA
            if use_rag and index_name.strip():
                if index_name not in rag.indexes:
                    st.warning(
                        f"Index '{index_name}' not found in memory. "
                        f"Please build it from PDFs in the sidebar."
                    )
                else:
                    st.info(f"[LOCAL LLaMA] Answering from index '{index_name}' ...")
                    t0 = time.time()
                    try:
                        rag_answer = rag.answer(index_name, question)
                        used_rag_flag = True
                        rag_time = time.time() - t0
                    except Exception as e:
                        st.error(f"RAG / LLaMA error: {e}")
                        rag_answer = None
                        used_rag_flag = False

            # 2) Gemini decision / reasoning system
            st.info("[GEMINI] Running decision / reasoning system...")

            if rag_answer:
                system_prompt = (
                    "You are a multi-agent style AI assistant specialising in "
                    f"{mode} tasks (engineering / policy / research).\n"
                    "You receive:\n"
                    "- The user's question\n"
                    "- A document-grounded answer generated by a local LLaMA RAG system "
                    "(over PDFs uploaded by the user).\n\n"
                    "Your job:\n"
                    "1) Treat the LLaMA answer as evidence / a first draft.\n"
                    "2) Analyse, refine, and extend it as needed.\n"
                    "3) Ensure the final answer is clear, structured, and aligned with the selected mode.\n"
                    "4) If something in the LLaMA answer seems wrong or incomplete, fix or augment it.\n"
                )
                user_content = (
                    f"MODE: {mode}\n\n"
                    f"USER QUESTION:\n{question}\n\n"
                    f"DOCUMENT-GROUNDED ANSWER (from local LLaMA RAG):\n{rag_answer}\n\n"
                    "Now produce your final improved answer."
                )
            else:
                system_prompt = (
                    "You are an expert AI assistant specialising in "
                    f"{mode} tasks (engineering / policy / research).\n"
                    "You have NO document context for this question.\n"
                    "Provide the best possible answer using your own knowledge, "
                    "structured, clear, and detailed."
                )
                user_content = question

            t1 = time.time()
            final_answer = call_gemini(system_prompt, user_content)
            gemini_time = time.time() - t1

            # 3) Evaluation (Gemini-based SCORE)
            eval_text = ""
            score = -1.0
            if auto_evaluate:
                st.info("[GEMINI] Evaluating answer quality (SCORE 0–10)...")
                t2 = time.time()
                try:
                    eval_text, score = evaluate_answer(
                        question=question,
                        final_answer=final_answer,
                        used_rag=used_rag_flag,
                        mode=mode,
                    )
                except Exception as e:
                    eval_text = f"[Evaluation error] {e}"
                    score = -1.0
                eval_time = time.time() - t2

            total_time = rag_time + gemini_time + eval_time
            word_count = len(final_answer.split())

            # Save interaction in history
            st.session_state.history.append(
                {
                    "mode": mode,
                    "question": question,
                    "use_rag": use_rag,
                    "index_name": index_name,
                    "rag_answer": rag_answer,
                    "final_answer": final_answer,
                    "eval_text": eval_text,
                    "score": score,
                    "rag_time": rag_time,
                    "gemini_time": gemini_time,
                    "eval_time": eval_time,
                    "total_time": total_time,
                    "word_count": word_count,
                }
            )

    # ---- Display history / latest interaction ----
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Conversation & Metrics")

        # Show the latest interaction on top
        for i, item in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            with st.expander(f"Interaction #{idx + 1}  –  mode={item['mode']}  (click to expand)", expanded=(i == 0)):
                st.markdown(f"**User question:**\n\n{item['question']}")

                if item["rag_answer"]:
                    tabs = st.tabs(["Final Answer (Gemini)", "PDF Answer (LLaMA RAG)", "Evaluation & Metrics"])
                else:
                    tabs = st.tabs(["Final Answer (Gemini)", "Evaluation & Metrics"])

                # Tab 1: final answer
                with tabs[0]:
                    st.markdown("### 🧠 Final Answer (Gemini)")
                    st.markdown(item["final_answer"])

                # Tab 2: raw PDF / RAG answer (if any)
                if item["rag_answer"]:
                    with tabs[1]:
                        st.markdown("### 📄 Document-Grounded Answer (local LLaMA RAG)")
                        st.markdown(item["rag_answer"])

                    eval_tab = tabs[2]
                else:
                    eval_tab = tabs[1]

                # Evaluation & metrics
                with eval_tab:
                    st.markdown("### 📊 Evaluation & System Metrics")

                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Quality SCORE (0–10)", f"{item['score']:.2f}" if item["score"] >= 0 else "N/A")
                    with cols[1]:
                        st.metric("Used PDF RAG?", "Yes" if item["use_rag"] and item["rag_answer"] else "No")
                    with cols[2]:
                        st.metric("Answer length (words)", item["word_count"])
                    with cols[3]:
                        st.metric("Total time (s)", f"{item['total_time']:.2f}")

                    st.markdown("#### ⏱ Timing breakdown")
                    st.write(
                        f"- RAG (local LLaMA) time: **{item['rag_time']:.2f} s**\n"
                        f"- Gemini reasoning time: **{item['gemini_time']:.2f} s**\n"
                        f"- Evaluation time: **{item['eval_time']:.2f} s**\n"
                    )

                    if item["eval_text"]:
                        st.markdown("#### 🧾 Evaluator feedback")
                        st.markdown(item["eval_text"])


if __name__ == "__main__":
    main()
