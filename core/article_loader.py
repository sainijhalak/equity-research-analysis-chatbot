import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from data.sample_articles import SAMPLE_FINANCIAL_ARTICLES

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def fetch_article_from_url(url: str) -> Dict[str, str]:
    """Scrapes news article text and title from a financial web URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        else:
            title = url

        # Extract main article paragraphs
        paragraphs = soup.find_all("p")
        text_content = "\n\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40])
        
        if not text_content or len(text_content) < 100:
            # Fallback for structured content
            body = soup.find("article") or soup.find("main") or soup.body
            if body:
                text_content = body.get_text(separator="\n", strip=True)

        return {
            "url": url,
            "title": title or "Financial News Document",
            "text": text_content or "No readable text content extracted from URL."
        }
    except Exception as e:
        return {
            "url": url,
            "title": f"Fetch Error ({url})",
            "text": f"Could not retrieve live content from URL: {str(e)}"
        }

def load_and_split_documents(urls: List[str] = None, use_sample_data: bool = False) -> List[Document]:
    """Loads financial news articles and splits them into chunks for FAISS."""
    articles = []

    if use_sample_data or not urls:
        articles.extend(SAMPLE_FINANCIAL_ARTICLES)

    if urls:
        for u in urls:
            u_clean = u.strip()
            if u_clean:
                # Check if it matches a sample URL
                matched_sample = next((s for s in SAMPLE_FINANCIAL_ARTICLES if s["url"] == u_clean), None)
                if matched_sample:
                    articles.append(matched_sample)
                else:
                    scraped = fetch_article_from_url(u_clean)
                    articles.append(scraped)

    # Convert to LangChain Documents
    raw_docs = []
    for art in articles:
        if art.get("text") and len(art["text"]) > 30:
            raw_docs.append(
                Document(
                    page_content=art["text"],
                    metadata={"source": art["url"], "title": art["title"]}
                )
            )

    # Split using LangChain RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    return chunks
