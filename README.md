# Classical NLP Smart Study Assistant

An interactive end-to-end study companion that implements **4 distinct classical (non-neural) NLP algorithms from scratch**, covering all 4 foundational NLP domains in a single unified pipeline:

| # | Classical Algorithm | NLP Problem Domain | Role in the Study Assistant |
| :--- | :--- | :--- | :--- |
| **1** | **TextRank (TF-IDF Cosine Graph + PageRank Power Iteration)** | **Text Summarization** | Builds a sentence similarity graph and solves for eigenvector centrality to generate an **Executive Chapter Summary**. |
| **2** | **Hidden Markov Model (HMM) with Viterbi Dynamic Programming** | **Information Extraction (IE)** | Tags sequences of tokens (`PERSON`, `LOCATION`, `DATE`, `CONCEPT`, `OTHER`) using transition $A_{i,j}$ and emission $B_{k,w}$ matrices decoded via the **Viterbi trellis** to extract structured entities and **Study Flashcards**. |
| **3** | **Okapi BM25 Probabilistic Ranking Model** | **Information Retrieval (IR)** | Indexes passages and ranks candidate evidence sentences against user questions using smooth Inverse Document Frequency ($IDF$), term saturation ($k_1$), and length normalization ($b$). |
| **4** | **Multinomial Naive Bayes EAT Classifier + Simplified Lesk WSD** | **Question & Answering (Q&A)** | Predicts the question's **Expected Answer Type** (`PERSON`, `LOCATION`, `DATE`, `CONCEPT`), pinpoints the exact answer phrase inside the BM25 passage using keyword proximity + HMM tags, and resolves polysemous word meanings via **Lesk dictionary-context overlap**. |

---

## Features
- **Works on Your Own Documents**: Upload any `.txt`, `.md`, or `.pdf` file, or paste custom lecture notes/articles directly in the UI.
- **Unified Single-Page Study Workspace**: Interactive annotated document reader on the left paired with the Executive Summary, Flashcards, and Q&A engine on the right.
- **Live Mathematical Inspector**: Click **"View Pipeline Math & Matrices"** in the top header to inspect the PageRank similarity/transition matrices, HMM transition matrix & Viterbi log-probability trellis, BM25 IDF tables, and Naive Bayes class posteriors.

---

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Flask web application**:
   ```bash
   python app.py
   ```
   Then open `http://127.0.0.1:5000` in your browser.
3. **Run the CLI numerical verification script**:
   ```bash
   python cli_demo.py
   ```
