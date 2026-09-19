# 🧠 Ultimate AI: Multi-Agent Visual RAG SaaS

An enterprise-grade, multi-tenant SaaS application built with Streamlit, LangChain, and Google Gemini. This platform moves far beyond standard semantic search by implementing a Multi-Agent Debate architecture, 3D Knowledge Graphs, real-time analytics, and secure multi-tenant isolation.

## 🌟 Enterprise SaaS Features

- **Multi-Tenant Architecture**: Complete data isolation. Every user gets their own dedicated vector database, user directory, and graph data to prevent data leakage.
- **Secure Authentication**: Built-in hashed login system powered by `streamlit-authenticator`. 
- **Admin Analytics Dashboard**: Real-time SaaS usage tracking (Total Users, Queries Today, MRR, AI Judge Scores) backed by a lightweight SQLite database.
- **Custom UI / UX**: A fully styled, responsive frontend featuring centered login cards, custom typography, sidebar profiles, and modern chat bubbles.

## 🤖 Advanced AI Capabilities

- **Multi-Agent Debate System**: A trio of specialized AI agents working together to answer user queries:
  1. *PDF Agent*: Scans local databases for document context.
  2. *Web Agent*: Searches the live internet (DuckDuckGo) for real-time data.
  3. *Manager Agent*: Synthesizes both reports, resolves contradictions, and delivers the final answer based on a dynamically customizable persona.
- **3D Knowledge Graphs**: Extracts entities and relationships from PDFs to render fully interactive, drag-and-drop 3D network visualizations using `PyVis`.
- **LLM-as-a-Judge & Self-Correction**: Uses an integrated Ragas-style evaluator to score its own answers. If the AI detects hallucinations (score < 0.75), it automatically forces a self-correction rewrite before showing the user.
- **Prompt Injection Firewall**: A dedicated Security Agent scans all user inputs for malicious jailbreaks or prompt injections before passing them to the main pipeline.
- **Hybrid Search & Local Reranking**: Fuses semantic vector search (ChromaDB) with keyword search (BM25), then reranks the results using a local HuggingFace Cross-Encoder.

## ⚙️ Tech Stack

- **Frontend**: Streamlit, Custom CSS
- **AI / Framework**: LangChain (Classic), Google Gemini (`gemini-3.6-flash`), DuckDuckGo
- **Databases**: ChromaDB (Vectors), SQLite (Usage Analytics & LLM Caching)
- **Visualizations**: PyVis (3D Graphs), Pandas & Streamlit Charts (Analytics)
- **Security**: Streamlit-Authenticator, bcrypt

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/rajeshbendi60-maker/AI-RAG-Evaluator.git
cd AI-RAG-Evaluator
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the application
```bash
python -m streamlit run app.py
```

### 4. Log in
To bypass the mock Stripe paywall and access the system, use the default administrator credentials:
- **Username**: `admin`
- **Password**: `abc`

## 🛠️ Deployment (CI/CD)
This project includes a `.github/workflows/deploy.yml` pipeline and a `Dockerfile` for seamless deployment to Google Cloud Run, AWS, or any Docker-compatible hosting environment.
