# Advanced RAG System

An intelligent, state-of-the-art Document Q&A chatbot built with Streamlit, LangChain, and the Google Gemini API. This application allows users to upload PDF documents and ask questions in a conversational interface, utilizing advanced retrieval techniques to ensure highly accurate, hallucination-free answers.

## Features

This project moves beyond standard semantic search by implementing a highly optimized, production-ready RAG (Retrieval-Augmented Generation) architecture:

- Conversational Memory: Remembers chat history for follow-up questions.
- Multi-Query Expansion: Uses an LLM to rewrite user queries into multiple variations, searching all of them simultaneously to capture broader context.
- Hybrid Search (Ensemble Retriever): Fuses standard vector search (ChromaDB) with keyword-based search (BM25) to capture both semantic meaning and exact terminology.
- Cross-Encoder Re-ranking: Uses a local lightweight HuggingFace model (ms-marco-MiniLM-L-6-v2) to strictly re-score and re-rank the retrieved chunks, ensuring the most relevant context is fed to the LLM.
- Custom LLM-as-a-Judge: Automatically evaluates its own answers on Faithfulness and Relevance without relying on heavy external metric libraries.

## Tech Stack
- Frontend: Streamlit
- Framework: LangChain
- LLM & Embeddings: Google Gemini (gemini-3.6-flash, gemini-embedding-001)
- Vector Database: ChromaDB
- Keyword Search: rank_bm25
- Local Re-ranker: sentence-transformers

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/rajeshbendi60-maker/AI-RAG-Evaluator.git
cd AI-RAG-Evaluator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
(Note: This project uses local Cross-Encoders which require PyTorch. The initial download may take a few minutes depending on your internet speed.)

### 3. Setup API Keys
Create a .env file in the root directory and add your Google Gemini API key:
```ini
GOOGLE_API_KEY=your_api_key_here
```
(Alternatively, you can input the API key directly in the app's sidebar).

### 4. Run the app
```bash
streamlit run app.py
```

## Usage
1. Open the local URL provided by Streamlit (usually http://localhost:8501).
2. Upload one or more PDF files via the sidebar.
3. Click "Process Documents" to chunk the text and build both the Chroma vector database and the BM25 keyword index.
4. Start chatting!
