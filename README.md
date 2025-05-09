---

# 🤗 [👉Hugging Face Deployed Streamlit App Link👈](https://huggingface.co/spaces/trohith89/ITC-Finance-Analyzer-NeuzenAI)

---

# 📊 ITC Financial Analyzer APP – AI-Powered Financial Q&A

This interactive AI tool dives into **ITC Ltd’s financial journey**, examining revenues, profitability, and fiscal performance through an intelligent chatbot interface. It fuses **smart web scraping**, **vector embeddings**, and **LLMs** to deliver chat-based, **data-backed answers with transparency**.



## 🌟 Key Capabilities

- 🤖 **Smart Data Scraper**: Fetches real-time financial disclosures using tools like *Tavily AI*.
- 🧬 **Embedding Engine**: Transforms scraped text into searchable vectors for high-relevance responses.
- 🗨️ **Conversational AI**: Ask natural questions, get reliable, structured replies grounded in ITC's actual data.
- 🧑‍💻 **Streamlit Interface**: A sleek web app for intuitive financial exploration.

---

## Project Structure

This repository follows a modular structure to separate different components of the project. Below is the breakdown of the project flow:

```bash
itc-financial-analysis/  
├── scraper/              # Tavily scripts for scraping financial data
├── database/             # Used ChromaDB for storing and processing data along with Embeddings
├── embeddings/           # Code for embedding generation and document chunking
├── llm/                  # Code for handling LLM queries and integration
├── app.py                # Streamlit UI for user interaction and Q&A
└── README.md             # Setup instructions, and usage details
```

## 📚 ITC Report Extractor Module

This backend component scrapes ITC Ltd's official reports and extracts the full text from PDFs using the **Tavily API**. Extracted documents are structured for downstream AI tasks using LangChain.

### 📌 Capabilities

- 🔄 Downloads & parses:
  - Annual Reports (2023–2024)
  - Quarterly Results (Q1–Q4, FY2023–FY2025)
  - Consolidated & Standalone Statements
- 🧠 Uses `extract_depth="advanced"` for deep content retrieval
- 🗂️ Adds clean metadata tags for traceability and AI reasoning

### 🧪 Dependencies

```bash
pip install tavily langchain

```
🧠 Embedding Layer with Chroma
This module preps ITC financial content into machine-understandable embeddings using GoogleGenerativeAIEmbeddings and stores them in a Chroma vector database for semantic search.

🔍 Functional Highlights
📥 Loads preprocessed LangChain documents (e.g., from pickle files)

✂️ Chunks long documents using RecursiveCharacterTextSplitter

🧠 Embeds using sentence-transformers/all-MiniLM-L6-v2

💾 Persists vectors in a local Chroma DB

🗜️ Zips the DB directory for easier reuse

📦 Install Dependencies

```
pip install langchain chromadb 

from langchain.embeddings import GoogleGenerativeAIEmbeddings

```

🧠 AI-Powered Q&A with Gemini
Engage in deep financial conversations using Google Gemini 2.0 Flash as the LLM. Combined with LangChain’s vector search, it retrieves the most relevant insights from ITC’s disclosures.

🔍 Features
📂 Uses MMR (Maximal Marginal Relevance) for sharp and diverse results

🧾 Cites source documents for full transparency

📊 Tailors answers to metrics, fiscal years, and company context


```
pip install streamlit langchain chromadb langchain-google-genai

```

💬 Streamlit Chat Interface – ITC Analyst
The main app enables interactive Q&A on ITC’s financials via a simple web chat. Responses are generated based on factual transcripts, with full traceability to the original data.

🎯 Objective
Answer queries about ITC’s earnings, margins, and key performance indicators

Maintain strict alignment with retrieved transcript data

Present year-wise breakdowns and financial facts in clear bullet formats

🧩 Core Tech Stack
🧱 Vector Search (Chroma + MMR) – Pulls relevant context chunks

🔮 LLM Reasoning (Gemini 2.0 Flash) – Converts data to insights

🧠 Chat Memory – Tracks the conversation thread within the session

⚙️ Getting Started
Place the zipped Chroma DB (e.g., CHROMA_DB_BACKUP.zip) in your working directory

Run the Streamlit app (app.py)

Ensure your Google Gemini API key is stored in .streamlit/secrets.toml like this:

```
GOOGLE_API_KEY = "your-api-key-here"

```
streamlit run app.py

```
git clone https://github.com/yourusername/repo-name.git
cd
```




Let me know if you’d like me to help generate a matching `app.py`, `requirements.txt`, or visuals like badges or workflow diagrams!



