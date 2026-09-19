import os
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


def get_chroma_path(username: str) -> str:
    return f"./chroma_db/{username}"

def get_bm25_path(username: str) -> str:
    return f"./bm25_retriever_{username}.pkl"

def get_embeddings_model():
    """Returns the Google Gemini embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_vector_store(chunks: List[Document], username: str = "default") -> Chroma:
    """
    Creates a Chroma vector database and a BM25 Keyword Index from the document chunks.
    """
    embeddings = get_embeddings_model()
    persist_directory = get_chroma_path(username)
    bm25_path = get_bm25_path(username)
    
    # Build and persist Chroma Vector Store
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    db.persist()
    print(f"Vector store built and persisted to {persist_directory}")
    
    # Build and persist BM25 Keyword Store
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("BM25 Keyword index built and persisted.")
        
    return db

def get_vector_store(username: str = "default") -> Chroma:
    """Loads an existing vector database."""
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=get_chroma_path(username), embedding_function=embeddings)

def get_advanced_retriever(db: Chroma, username: str = "default"):
    """
    Returns a State-of-the-Art Advanced Retriever combining:
    1. Multi-Query Expansion
    2. Hybrid Search (Vector + Keyword BM25)
    3. Cross-Encoder Re-ranking
    """
    # 1. Base Vector Retriever
    vector_retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # 2. Multi-Query Retriever (Uses 1 LLM call to generate 3 variations of the query)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_retriever,
        llm=llm
    )
    
    # 3. BM25 Keyword Retriever
    bm25_path = get_bm25_path(username)
    try:
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
            bm25_retriever.k = 10
    except FileNotFoundError:
        print("BM25 index not found. Falling back to Vector-only retrieval.")
        bm25_retriever = None
        
    # 4. Hybrid Search (Ensemble)
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, multi_query_retriever],
            weights=[0.5, 0.5]
        )
        retriever_to_rerank = ensemble_retriever
    else:
        retriever_to_rerank = multi_query_retriever
        
    # 5. Cross-Encoder Re-ranking (Local Model, No API limits!)
    # We retrieve more documents (e.g., 20) and have the cross-encoder pick the top 5
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever_to_rerank
    )
    
    return compression_retriever
