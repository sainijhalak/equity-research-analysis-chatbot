import os
import shutil
from typing import List, Tuple, Any
from pathlib import Path
from langchain_core.documents import Document
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    FAISS = Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from config import FAISS_INDEX_DIR, DEFAULT_EMBEDDING_MODEL

class SimpleInMemoryVectorStore:
    """Lightweight in-memory vector store for serverless environments."""
    def __init__(self, documents: List[Document], embeddings: Embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.doc_vectors = [embeddings.embed_query(d.page_content) for d in documents]

    def as_retriever(self, search_kwargs: dict = None):
        k = search_kwargs.get("k", 4) if search_kwargs else 4
        class SimpleRetriever:
            def __init__(self, store, top_k):
                self.store = store
                self.top_k = top_k
            def invoke(self, query: str) -> List[Document]:
                q_vec = self.store.embeddings.embed_query(query)
                scored = []
                for doc, d_vec in zip(self.store.documents, self.store.doc_vectors):
                    # Cosine dot product
                    score = sum(a * b for a, b in zip(q_vec, d_vec))
                    scored.append((score, doc))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored[:self.top_k]]
        return SimpleRetriever(self, k)

class FallbackDemoEmbeddings(Embeddings):
    """Deterministic 64-dimensional bag-of-words embeddings for offline interview demos."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        import hashlib
        vec = [0.0] * 64
        words = text.lower().split()
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % 64
            vec[idx] += 1.0
        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

def get_embeddings_model(api_key: str = None) -> Tuple[Embeddings, bool]:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if key.strip():
        try:
            return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, openai_api_key=key.strip()), True
        except Exception:
            pass
    return FallbackDemoEmbeddings(), False

def build_faiss_vectorstore(documents: List[Document], api_key: str = None) -> Tuple[Any, bool]:
    """Builds a FAISS vector store from LangChain Document chunks, with in-memory fallback."""
    embeddings, is_openai = get_embeddings_model(api_key)
    
    if FAISS_AVAILABLE:
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            try:
                save_faiss_vectorstore(vectorstore)
            except Exception:
                pass
            return vectorstore, is_openai
        except Exception:
            pass

    # Serverless in-memory vector store fallback
    return SimpleInMemoryVectorStore(documents, embeddings), is_openai

def save_faiss_vectorstore(vectorstore: Any, folder_path: Path = FAISS_INDEX_DIR):
    folder_path.mkdir(parents=True, exist_ok=True)
    if hasattr(vectorstore, "save_local"):
        vectorstore.save_local(str(folder_path))

def load_faiss_vectorstore(api_key: str = None, folder_path: Path = FAISS_INDEX_DIR) -> Any:
    if not FAISS_AVAILABLE:
        return None
    embeddings, _ = get_embeddings_model(api_key)
    if (folder_path / "index.faiss").exists():
        return FAISS.load_local(str(folder_path), embeddings, allow_dangerous_deserialization=True)
    return None
