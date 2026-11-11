import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.article_loader import load_and_split_documents
from core.vector_store import build_faiss_vectorstore
from core.rag_chain import answer_financial_query

class TestEquityResearchRAG(unittest.TestCase):
    def test_document_chunking(self):
        docs = load_and_split_documents([], use_sample_data=True)
        self.assertGreater(len(docs), 0)
        self.assertIn("source", docs[0].metadata)

    def test_faiss_indexing_and_query(self):
        docs = load_and_split_documents([], use_sample_data=True)
        vs, is_openai = build_faiss_vectorstore(docs)
        self.assertIsNotNone(vs)
        ans = answer_financial_query("What was Nvidia's revenue?", vs)
        self.assertIn("answer", ans)
        self.assertGreater(len(ans["sources"]), 0)

if __name__ == "__main__":
    unittest.main()
