import streamlit as st
import time
from llm_client import call_gemini
from rag_engine import RAGEngine
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import tempfile

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ------------------------ METRICS ------------------------
def compute_metrics(question, rag_answer, solver, evaluator, improver, rag_time, gemini_time):
    metrics = {}

    # Precision = how well retrieved text matches question
    if rag_answer:
        metrics["Precision Score"] = cosine(
            embedder.encode(question),
            embedder.encode(rag_answer)
        )
    else:
        metrics["Precision Score"] = "N/A"

    # Recall = how much retrieved text matches final answer
    if rag_answer:
        metrics["Recall Score"] = cosine(
            embedder.encode(improver),
            embedder.encode(rag_answer)
        )
    else:
        metrics["Recall Score"] = "N/A"

    # Alignment Score
    try:
        metrics["Answer–Document Alignment"] = cosine(
            embedder.encode(improver),
            embedder.encode(rag_answer)
        ) if rag_answer else "N/A"
    except:
        metrics["Answer–Document Alignment"] = "N/A"

    # Coherence
    try:
        metrics["Coherence Score"] = cosine(
            embedder.encode(evaluator),
            embedder.encode(improver)
        )
    except:
        metrics["Coherence Score"] = "N/A"

    # Groundedness
    if rag_answer:
        metrics["Groundedness Score"] = cosine(
            embedder.encode(improver),
            embedder.encode(rag_answer)
        )
    else:
        metrics["Groundedness Score"] = "N/A"

    # Novelty (1 - groundedness)
    if rag_answer:
        metrics["Novelty Score"] = 1 - metrics["Groundedness Score"]
    else:
        metrics["Novelty Score"] = "N/A"

    # Latency
    metrics["RAG Time (s)"] = rag_time
    metrics["Gemini Time (s)"] = gemini_time
    metrics["Total Time (s)"] = rag_time + gemini_time

    return metrics


# ------------------------ STREAMLIT UI ------------------------
st.set_page_config(page_title="Hybrid AI System", layout="wide")

st.markdown("## 🚀 Autonomous Hybrid AI System (Gemini + LLaMA RAG)")

# Simple dropdown
mode = st.selectbox("Choose Mode:", ["engineering", "policy", "research"])

# Question input
question = st.text_area("Enter your question:")

# Optional PDF upload button (+)
uploaded_pdfs = st.file_uploader("Upload PDFs (Optional)", type=["pdf"], accept_multiple_files=True)

# Run button
run = st.button("⚡ Run System")

if run:
    rag_answer = ""
    rag_time = 0

    # ------------------ HANDLE PDF IF UPLOADED ------------------
    if uploaded_pdfs:
        st.info("Processing uploaded PDFs with Local LLaMA (RAG)...")

        tmp_dir = tempfile.mkdtemp()
        pdf_paths = []

        for pdf in uploaded_pdfs:
            save_path = os.path.join(tmp_dir, pdf.name)
            with open(save_path, "wb") as f:
                f.write(pdf.read())
            pdf_paths.append(save_path)

        rag = RAGEngine()

        t0 = time.time()
        rag.build_index("uploaded_docs", tmp_dir)
        rag_answer = rag.answer("uploaded_docs", question)
        rag_time = time.time() - t0

        st.subheader("📄 PDF-grounded Answer (LLaMA RAG)")
        st.write(rag_answer if rag_answer else "No relevant text found.")

        context = f"DOCUMENT CONTEXT:\n{rag_answer}\n"

    else:
        context = ""

    # ------------------ GEMINI MULTI-AGENT REASONING ------------------
    st.info("Running Gemini multi-agent reasoning...")

    t1 = time.time()

    solver = call_gemini(f"You are a {mode} expert. Answer clearly.", question + "\n" + context)
    evaluator = call_gemini("Evaluate and refine the answer.", solver)
    improver = call_gemini("Improve using evaluator feedback.", evaluator)

    gemini_time = time.time() - t1

    # ------------------ FINAL ANSWER ------------------
    st.subheader("🧠 Final Answer (Gemini Improver)")
    st.write(improver)

    # ------------------ METRICS ------------------
    metrics = compute_metrics(
        question, rag_answer, solver, evaluator, improver, rag_time, gemini_time
    )

    st.subheader("📊 Evaluation Metrics")

    col1, col2 = st.columns(2)
    with col1:
        for key in ["Precision Score", "Recall Score", "Answer–Document Alignment", "Coherence Score"]:
            st.metric(key, metrics[key])
    with col2:
        for key in ["Groundedness Score", "Novelty Score", "RAG Time (s)", "Gemini Time (s)"]:
            st.metric(key, metrics[key])

    st.metric("Total Time (s)", metrics["Total Time (s)"])

    st.subheader("🧩 Raw Metrics JSON")
    st.json(metrics)
