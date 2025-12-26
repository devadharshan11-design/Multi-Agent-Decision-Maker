# 🚀 Autonomous Hybrid Multi-Agent GenAI System  
### (Gemini + Local LLaMA RAG with Evaluation Metrics)

This project implements an **Autonomous Hybrid Generative AI system** that combines:

- **Local document-grounded RAG (Retrieval Augmented Generation)** using **LLaMA via Ollama**
- **Cloud-based multi-agent reasoning** using **Google Gemini**
- **Quantitative evaluation metrics** to objectively assess answer quality, grounding, coherence, and latency

The system is designed to work **both with and without PDFs**, automatically adapting its reasoning pipeline.


📌 Key Highlights

🔹 Hybrid architecture: **Local + Cloud LLMs**
🔹 Multi-agent reasoning: **Solver → Evaluator → Improver**
🔹 Document-grounded answers using **PDF-RAG**
🔹 Fully **interactive Streamlit UI**
🔹 Real-time **evaluation metrics dashboard**
🔹 Secure **environment-based API key handling**

---

🧠 System Architecture

### 1. Local RAG (LLaMA via Ollama)
- PDFs are chunked and embedded locally
- Relevant chunks are retrieved using semantic similarity
- Answers are generated **strictly grounded in documents**

### 2. Gemini Multi-Agent Pipeline
When Gemini is used, it operates as **multiple logical agents**:
- **Solver** – generates the initial answer
- **Evaluator** – critiques correctness, depth, and alignment
- **Improver** – produces the final refined answer

### 3. Adaptive Execution
- ✅ If PDF is uploaded → **RAG + Gemini**
- ✅ If no PDF → **Gemini only**
- Metrics adapt automatically based on availability of documents

---

## 📊 Evaluation Metrics Implemented

The system computes the following **quantitative metrics**:

| Metric | Description |
|------|------------|
| Precision Score | Semantic relevance of generated answer |
| Recall Score | Coverage of document information |
| Answer–Document Alignment | Similarity between answer and retrieved context |
| Coherence Score | Logical consistency of final response |
| Groundedness Score | Degree of reliance on document content |
| Novelty Score | Measure of non-redundant reasoning |
| RAG Time (s) | Time spent in local document retrieval |
| Gemini Time (s) | Cloud reasoning time |
| Total Time (s) | End-to-end latency |

> **Note:** When no PDF is used, document-dependent metrics are shown as `N/A` by design.

---

## 🖥️ User Interface

- Built using **Streamlit**
- Minimal, clean interaction flow:
  1. Select mode (engineering / research / policy)
  2. Enter question
  3. Optionally upload PDF(s)
  4. Click **Run**
- Results displayed in **separate tabs**:
  - Final Answer
  - PDF-Grounded Answer
  - Evaluation Metrics

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone  https://github.com/devadharshan11-design/Multi-Agent-Decision-Maker
cd Multi-Agent-Decision-Maker
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install Ollama (for Local LLaMA)

👉 [https://ollama.com/download](https://ollama.com/download)

Pull model:

```bash
ollama pull llama3.1
```

Ensure Ollama is running:

```bash
ollama serve
```

---

## 🔐 Environment Setup (IMPORTANT)

Create a `.env` file **locally** (do NOT push to GitHub):

```
GEMINI_API_KEY=your_api_key_here
```

The project uses **python-dotenv** to securely load credentials.

---

## ▶️ Run the Application

```bash
streamlit run ui_app.py
```

Then open:

```
http://localhost:8501
```

---

## 🛡️ Security Practices

* API keys are **never hard-coded**
* `.env` files are excluded via `.gitignore`
* Secret exposure detection handled via GitHub Secret Scanning
* Keys can be rotated without code changes

---

## 🎯 Applications

* Research paper summarization
* Technical document analysis
* Policy and engineering Q&A
* AI evaluation and benchmarking
* Demonstration of autonomous GenAI systems

---

## 📈 Project Value

This project demonstrates:

* Advanced **GenAI system design**
* Real-world **RAG + evaluation**
* Practical **AI safety and grounding**
* Professional-grade **engineering practices**

---


---

## 📜 License

This project is intended for **academic and educational use**.

---

> ⚠️ Note: Ensure Ollama is running before enabling PDF-RAG.
> If Gemini quota is exceeded, the system will still function in **local RAG mode**.



