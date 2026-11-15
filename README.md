# 📈 Equity Research News Analysis Chatbot

[![Vercel](https://img.shields.io/badge/Deployed-Vercel%20Serverless-000000.svg?logo=vercel)](https://equity-research-analysis.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-orange.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-00599C.svg)](https://github.com/facebookresearch/faiss)
[![OpenAI API](https://img.shields.io/badge/OpenAI%20API-GPT--4o%20%7C%20Embeddings-412991.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **Equity Research News Analysis Chatbot** is an AI-powered financial intelligence tool engineered to solve information overload for investment analysts. It scrapes, chunks, indexes, and queries multiple financial news articles and earnings releases, delivering context-aware, quantitatively grounded, and source-backed answers in seconds.

---

## 🌟 Key Features & Engineering Highlights

- 📰 **Multi-URL News Ingestion**: Ingests articles from Bloomberg, Reuters, Yahoo Finance, and CNBC. Scrapes HTML, sanitizes boilerplate, and extracts clean financial reporting.
- ✂️ **Deterministic Semantic Chunking**: Employs LangChain's `RecursiveCharacterTextSplitter` (chunk size: 1000 characters, overlap: 200) to preserve context continuity across financial balance sheets and management commentary.
- ⚡ **High-Speed In-Memory Vector Search (FAISS)**: Uses Facebook AI Similarity Search (FAISS) coupled with OpenAI's `text-embedding-3-small` for sub-millisecond semantic similarity retrieval.
- 🎯 **Source-Grounded Financial Q&A (RAG)**: LangChain LCEL pipeline engineered with strict prompts preventing hallucinations:
  - Answers cite exact source article titles and clickable URLs.
  - Prioritizes quantitative figures: YoY revenue growth, EPS, gross margins, and forward guidance.
  - Transparently acknowledges lack of context if news does not contain requested datapoints.
- 💼 **Analyst Dashboard**: Modern Streamlit interface with 1-click curated tech articles (Nvidia, Apple, Tesla), pre-set analyst question chips, metric summaries, and raw chunk inspection.

---

## 🏗️ System Architecture

```
  [ Financial News URLs ]  ──>  [ Scraper (BS4 / Requests) ]
                                            │
                                            ▼
                               [ LangChain Document Chunker ]
                                (RecursiveCharacterTextSplitter)
                                            │
                                            ▼
                               [ Vector Embedding Engine ]
                                (text-embedding-3-small)
                                            │
                                            ▼
                                   [ FAISS Vector Store ]
                                            │
    [ User / Analyst Query ]                │
               │                            │
               ▼                            ▼
        [ Vector Query ]  ───────>  [ Top-K Similar Chunks ]
                                            │
                                            ▼
                                  [ Financial RAG Prompt ]
                                            │
                                            ▼
                               [ OpenAI GPT-4o / Fallback ]
                                            │
                                            ▼
                              [ Context-Aware Answer + Citations ]
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sainijhalak/equity-research-analysis-chatbot.git
cd equity-research-analysis-chatbot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure OpenAI Key (Optional)
```bash
cp .env.example .env
```
```env
OPENAI_API_KEY=sk-proj-your-openai-api-key-here
```
> *Note: If an OpenAI API key is not supplied, the app gracefully activates its built-in offline demo engine, performing semantic extraction and displaying verified citations without throwing errors.*

### 3. Launch Analyst Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 Technical Interview Talking Points

1. **Why FAISS for Financial News Search?**  
   FAISS provides blazing-fast in-memory and disk-persisted vector indexing. For equity analysts analyzing 5–50 articles per stock ticker, FAISS provides near-zero latency retrieval without cloud database costs.
2. **Handling Hallucinations in Financial Analysis:**  
   In financial research, hallucinating a revenue figure or EBITDA margin is catastrophic. Our system prompt enforces strict refusal: if a metric is not present in the retrieved chunks, the assistant explicitly states *"insufficient data to answer"* rather than guessing.
3. **Chunking Strategy for Earnings Releases:**  
   Financial reports interleave quantitative tables with qualitative narrative. Using a 1000-character chunk with a 200-character overlap prevents cutting crucial guidance figures across chunk boundaries.

---

## 📄 License
MIT License. Created by Jhalak Saini.
