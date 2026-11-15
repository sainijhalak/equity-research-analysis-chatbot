import json
import os
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.article_loader import load_and_split_documents
from core.vector_store import build_faiss_vectorstore
from core.rag_chain import answer_financial_query
from data.sample_articles import SAMPLE_FINANCIAL_ARTICLES

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if self.path.endswith("/sample"):
            self.wfile.write(json.dumps({
                "status": "success",
                "articles": SAMPLE_FINANCIAL_ARTICLES
            }).encode("utf-8"))
        else:
            self.wfile.write(json.dumps({
                "status": "ok",
                "service": "Equity Research Serverless RAG API",
                "version": "1.0.0"
            }).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}

        question = data.get("question", "").strip()
        urls = data.get("urls", [])
        api_key = data.get("api_key", "").strip()

        if not question:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "message": "Missing required field 'question'"
            }).encode("utf-8"))
            return

        try:
            # Load and split documents
            docs = load_and_split_documents(urls=urls, use_sample_data=(len(urls) == 0))
            
            # Build vector store
            vectorstore, _ = build_faiss_vectorstore(docs, api_key=api_key)

            # Query RAG chain
            res = answer_financial_query(question, vectorstore, api_key=api_key)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "question": question,
                "answer": res.get("answer", ""),
                "sources": res.get("sources", []),
                "is_live_llm": res.get("is_live_llm", False),
                "chunks_count": len(docs)
            }).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "message": f"Serverless RAG Error: {str(e)}"
            }).encode("utf-8"))
