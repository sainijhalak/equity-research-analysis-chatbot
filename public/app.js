/**
 * Equity Research AI – Serverless Client
 */

const sampleArticles = [
    {
        url: "https://finance.yahoo.com/news/nvidia-q2-earnings-blackwell-demand-record-revenue",
        title: "NVIDIA Reports Record Q2 Revenue Driven by Unprecedented Blackwell GPU Demand",
        text: "SANTA CLARA, CA — NVIDIA Corporation reported record revenue of $30.04 billion, up 122% YoY. Data Center revenue reached $26.3 billion (+154% YoY). Gross margin was 75.1%."
    },
    {
        url: "https://www.reuters.com/technology/apple-q3-services-hit-all-time-high-amid-ai-rollout",
        title: "Apple Q3 Results Top Wall Street Expectations on Record Services Growth",
        text: "CUPERTINO, CA — Apple posted quarterly revenue of $85.8 billion (+5% YoY). Services reached an all-time high of $24.2 billion (+14.1% YoY). Diluted EPS was $1.40 (+11%)."
    },
    {
        url: "https://www.bloomberg.com/news/articles/tesla-robotaxi-q2-energy-storage-margin-expansion",
        title: "Tesla Q2 Financial Highlights: Energy Storage Surge and Autonomous Fleet Strategy",
        text: "AUSTIN, TX — Tesla revenues grew to $25.5 billion (+2% YoY). Energy storage deployed a record 9.4 GWh (+157% YoY) with expanding gross margins. Free cash flow recovered to $1.34B."
    }
];

function renderArticles() {
    const list = document.getElementById("articlesList");
    if (!list) return;
    list.innerHTML = "";
    sampleArticles.forEach((art, idx) => {
        const item = document.createElement("div");
        item.className = "article-item";
        item.innerHTML = `
            <h4>📄 Article ${idx + 1}: ${art.title}</h4>
            <p><strong>Source URL:</strong> <a href="${art.url}" target="_blank" style="color: #38bdf8;">${art.url}</a></p>
            <p style="margin-top: 0.5rem;">${art.text}</p>
        `;
        list.appendChild(item);
    });
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

    const targetTab = document.getElementById(tabId);
    if (targetTab) targetTab.classList.add("active");

    const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick").includes(tabId));
    if (activeBtn) activeBtn.classList.add("active");
}

function resetToCurated() {
    document.getElementById("url1").value = sampleArticles[0].url;
    document.getElementById("url2").value = sampleArticles[1].url;
    document.getElementById("url3").value = sampleArticles[2].url;
}

function askPreset(question) {
    document.getElementById("questionInput").value = question;
    switchTab("chatTab");
    sendQuery();
}

async function sendQuery() {
    const input = document.getElementById("questionInput");
    const q = input.value.trim();
    if (!q) return;

    const apiKey = document.getElementById("apiKeyInput").value.trim();
    const url1 = document.getElementById("url1").value.trim();
    const url2 = document.getElementById("url2").value.trim();
    const url3 = document.getElementById("url3").value.trim();
    const urls = [url1, url2, url3].filter(Boolean);

    const container = document.getElementById("chatContainer");

    // Append User Message
    const userMsg = document.createElement("div");
    userMsg.className = "message user-message";
    userMsg.innerHTML = `<div class="message-header"><strong>Analyst Query</strong></div><div>${escapeHtml(q)}</div>`;
    container.appendChild(userMsg);
    input.value = "";

    // Append Loading Assistant Message
    const loadingMsg = document.createElement("div");
    loadingMsg.className = "message assistant-message";
    loadingMsg.innerHTML = `<div class="message-header"><strong>Equity Research Analyst AI</strong></div><div><em>⚡ Querying LangChain serverless RAG pipeline & vector index...</em></div>`;
    container.appendChild(loadingMsg);
    container.scrollTop = container.scrollHeight;

    const sendBtn = document.getElementById("sendBtn");
    sendBtn.disabled = true;

    try {
        const response = await fetch("/api", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q, urls: urls, api_key: apiKey })
        });

        const data = await response.json();

        if (data.status === "success") {
            let sourcesHtml = "";
            if (data.sources && data.sources.length > 0) {
                sourcesHtml = `<div class="citation-card"><strong>🔗 Verified Citations:</strong><ul style="margin-top: 0.3rem; padding-left: 1.2rem;">`;
                data.sources.forEach(s => {
                    sourcesHtml += `<li><a href="${escapeHtml(s.source)}" target="_blank">${escapeHtml(s.title || s.source)}</a></li>`;
                });
                sourcesHtml += `</ul></div>`;
            }

            loadingMsg.innerHTML = `
                <div class="message-header">
                    <strong>Equity Research Analyst AI</strong> 
                    <span style="font-size: 0.75rem; color: #10b981; margin-left: 0.5rem;">(${data.chunks_count || 6} chunks evaluated)</span>
                </div>
                <div style="white-space: pre-line;">${formatMarkdown(data.answer)}</div>
                ${sourcesHtml}
            `;
        } else {
            loadingMsg.innerHTML = `
                <div class="message-header"><strong>System Notice</strong></div>
                <div style="color: #ef4444;">${escapeHtml(data.message || "Failed to process query.")}</div>
            `;
        }
    } catch (err) {
        loadingMsg.innerHTML = `
            <div class="message-header"><strong>Connection Error</strong></div>
            <div style="color: #ef4444;">Could not reach serverless API: ${escapeHtml(err.message)}</div>
        `;
    } finally {
        sendBtn.disabled = false;
        container.scrollTop = container.scrollHeight;
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.innerText = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return "";
    return text
        .replace(/### (.*?)\n/g, '<h4 style="color:#38bdf8; margin: 0.5rem 0;">$1</h4>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

document.addEventListener("DOMContentLoaded", () => {
    renderArticles();
});
