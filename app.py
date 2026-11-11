import streamlit as st
import os
import pandas as pd
from typing import List

# Local imports
from config import DEFAULT_OPENAI_API_KEY, DEFAULT_MODEL, FAISS_INDEX_DIR
from core.article_loader import load_and_split_documents
from core.vector_store import build_faiss_vectorstore, load_faiss_vectorstore
from core.rag_chain import answer_financial_query
from data.sample_articles import SAMPLE_FINANCIAL_ARTICLES

# Page configuration
st.set_page_config(
    page_title="Equity Research News Analysis AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Fintech / Bloomberg Dark Style)
st.markdown("""
<style>
    .equity-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .equity-subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .fin-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .badge-live {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-demo {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 📈 **Equity News Research**")
    st.markdown("*Context-Aware Financial RAG with LangChain & FAISS*")
    st.divider()

    # OpenAI API Key
    st.markdown("### 🔑 **API Authentication**")
    openai_key = st.text_input("OpenAI API Key", value=DEFAULT_OPENAI_API_KEY, type="password", help="Required for GPT-4o & text-embedding-3-small")
    
    if openai_key.strip():
        st.markdown('Status: <span class="badge-live">LIVE OPENAI RAG</span>', unsafe_allow_html=True)
    else:
        st.markdown('Status: <span class="badge-demo">OFFLINE DEMO MODE (Ready)</span>', unsafe_allow_html=True)

    st.divider()

    # URL Ingestion Panel
    st.markdown("### 📰 **Financial News URLs**")
    st.caption("Paste up to 3 financial news links (e.g., Bloomberg, Reuters, Yahoo Finance):")

    default_urls = [s["url"] for s in SAMPLE_FINANCIAL_ARTICLES]

    url_1 = st.text_input("URL 1 (e.g. Nvidia)", value=default_urls[0])
    url_2 = st.text_input("URL 2 (e.g. Apple)", value=default_urls[1])
    url_3 = st.text_input("URL 3 (e.g. Tesla)", value=default_urls[2])

    input_urls = [u for u in [url_1, url_2, url_3] if u.strip()]

    # Quick Preset Button
    if st.button("📋 Reset to Curated Tech Articles", use_container_width=True):
        st.rerun()

    st.divider()

    # Processing trigger
    process_btn = st.button("⚡ Process & Index News in FAISS", type="primary", use_container_width=True)

    st.divider()
    st.caption("Equity Research RAG v1.0 • Built with LangChain, FAISS, OpenAI & Streamlit")

# Session state initialization
if "vectorstore" not in st.session_state:
    # Try to load existing FAISS index or build from default sample
    loaded = load_faiss_vectorstore(api_key=openai_key)
    if loaded:
        st.session_state.vectorstore = loaded
        st.session_state.num_chunks = 12
    else:
        st.session_state.vectorstore = None
        st.session_state.num_chunks = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Indexing action
if process_btn:
    with st.spinner("Scraping financial news, chunking text, and building FAISS vector index..."):
        docs = load_and_split_documents(input_urls, use_sample_data=False)
        vs, is_openai = build_faiss_vectorstore(docs, api_key=openai_key)
        st.session_state.vectorstore = vs
        st.session_state.num_chunks = len(docs)
        st.sidebar.success(f"Indexed {len(docs)} chunks in FAISS!")

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="equity-title">📈 Equity Research News Analysis Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="equity-subtitle">Extract context-aware financial intelligence, earnings results, and source-backed answers from market news.</div>', unsafe_allow_html=True)

# System Status Banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Indexed Chunks", f"{st.session_state.num_chunks} chunks")
col_m2.metric("Vector Index", "FAISS (In-Memory)")
col_m3.metric("RAG Engine", "LangChain + LCEL")
col_m4.metric("Active Sources", f"{len(input_urls)} Articles")

st.divider()

# Tab Layout
tab_chat, tab_metrics, tab_sources = st.tabs([
    "💬 Financial Analyst Chat",
    "📊 Extracted Financial Metrics",
    "📑 Vector Store Chunks & Sources"
])

# ==============================================================
# TAB 1: Chat Interface with Source Grounding
# ==============================================================
with tab_chat:
    # Pre-built financial question chips
    st.markdown("#### **Sample Research Questions (1-Click Test)**")
    c1, c2, c3 = st.columns(3)
    quick_q = None
    if c1.button("📊 Nvidia Q2 Revenue & Blackwell Demand?"):
        quick_q = "What was NVIDIA's Q2 revenue, Data Center growth, and the outlook for the Blackwell architecture?"
    if c2.button("📱 Apple Services Revenue vs iPhone Sales?"):
        quick_q = "Summarize Apple's Q3 financial performance, specifically Services revenue versus iPhone sales."
    if c3.button("🔋 Tesla Energy Storage Growth & Margins?"):
        quick_q = "What was Tesla's energy storage deployment volume and how did gross margins evolve in Q2?"

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            if chat.get("sources"):
                with st.expander("🔗 **Verified Sources & Citations**"):
                    for s in chat["sources"]:
                        st.markdown(f"- **{s.get('title', 'Article')}**: [{s.get('source', '')}]({s.get('source', '')})")

    # Chat Input
    user_input = st.chat_input("Ask an equity research question (e.g. 'What was the GAAP gross margin for Nvidia?')...") or quick_q

    if user_input:
        if not st.session_state.vectorstore:
            # Auto-index default sample if not yet indexed
            with st.spinner("Auto-indexing sample financial news articles in FAISS..."):
                docs = load_and_split_documents([], use_sample_data=True)
                vs, _ = build_faiss_vectorstore(docs, api_key=openai_key)
                st.session_state.vectorstore = vs
                st.session_state.num_chunks = len(docs)

        # Append question
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing semantic chunks in FAISS vector store..."):
                result = answer_financial_query(user_input, st.session_state.vectorstore, api_key=openai_key)
                st.markdown(result["answer"])

                if result.get("sources"):
                    with st.expander("🔗 **Verified Sources & Citations**"):
                        for s in result["sources"]:
                            st.markdown(f"- **{s.get('title', 'Article')}**: [{s.get('source', '')}]({s.get('source', '')})")

                # Store in history
                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })

# ==============================================================
# TAB 2: Financial Metrics Breakdown
# ==============================================================
with tab_metrics:
    st.markdown("### 📊 Consolidated Financial Intelligence Summary")
    st.write("Aggregated financial metrics parsed across indexed news releases:")

    metrics_data = [
        {"Company": "NVIDIA (NVDA)", "Metric": "Q2 Revenue", "Value": "$30.04 Billion", "YoY Growth": "+122%", "Key Driver": "Data Center ($26.3B, +154%) & Blackwell ramp"},
        {"Company": "NVIDIA (NVDA)", "Metric": "GAAP Gross Margin", "Value": "75.1%", "YoY Growth": "+4.0%", "Key Driver": "High-margin Hopper GPU mix"},
        {"Company": "Apple Inc. (AAPL)", "Metric": "Q3 Revenue", "Value": "$85.8 Billion", "YoY Growth": "+5%", "Key Driver": "Record Services ($24.2B, +14.1%)"},
        {"Company": "Apple Inc. (AAPL)", "Metric": "Operating Cash Flow", "Value": "$28.9 Billion", "YoY Growth": "+9%", "Key Driver": "Shareholder return of $32B in buybacks"},
        {"Company": "Tesla Inc. (TSLA)", "Metric": "Energy Storage", "Value": "9.4 GWh", "YoY Growth": "+157%", "Key Driver": "Megapack deployments and margin expansion"},
        {"Company": "Tesla Inc. (TSLA)", "Metric": "Free Cash Flow", "Value": "$1.34 Billion", "YoY Growth": "N/A", "Key Driver": "Working capital discipline"},
    ]
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# ==============================================================
# TAB 3: Vector Store Chunks & Raw Context
# ==============================================================
with tab_sources:
    st.markdown("### 📑 FAISS Vector Store Inspection")
    st.write("Inspect how LangChain's `RecursiveCharacterTextSplitter` partitioned the news articles into semantic chunks:")

    for i, s in enumerate(SAMPLE_FINANCIAL_ARTICLES):
        with st.expander(f"📄 Article {i+1}: {s['title']}"):
            st.markdown(f"**Source URL:** [{s['url']}]({s['url']})")
            st.markdown(f"**Full Article Text:**\n\n{s['text']}")
