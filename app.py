import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from data_ingestion import chunk_documents
from langchain_core.documents import Document
from retriever import build_vector_store, get_vector_store, get_advanced_retriever
from evaluator import evaluate_rag_response

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Advanced RAG Evaluator", layout="wide")
st.title("🧠 Advanced RAG System with Chat History")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not api_key:
            st.error("Please enter a Google API Key first.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Processing, Chunking, and Building Dual-Indexes (Vector + BM25)..."):
                # Save uploaded files temporarily
                os.makedirs("./data", exist_ok=True)
                from langchain_community.document_loaders import PyPDFLoader
                docs = []
                for file in uploaded_files:
                    file_path = os.path.join("./data", file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                
                # Chunk and build vector store
                chunks = chunk_documents(docs)
                st.write(f"Created {len(chunks)} chunks.")
                
                build_vector_store(chunks)
                # Clear chat history on new document upload
                st.session_state.chat_history = []
                st.success("Vector DB & Keyword Index Built Successfully!")

# --- Main Chat Interface ---
st.subheader("2. Chat with your Documents")

# Display Chat History
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat Input
user_query = st.chat_input("Ask a question about the documents...")

if user_query:
    if not api_key:
        st.error("API Key required.")
        st.stop()
        
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Retrieving via Hybrid Search & Cross-Encoder..."):
            try:
                db = get_vector_store()
                retriever = get_advanced_retriever(db)
                llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
                
                # 1. History Aware Retriever Prompt (Rewrites the query)
                contextualize_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                
                # 2. QA Prompt
                system_prompt = (
                    "You are an expert assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "Keep the answer concise.\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                contexts = [doc.page_content for doc in response["context"]]
                
                st.markdown(answer)
                
                with st.expander("View Retrieved Contexts (Top 5 Reranked)"):
                    for i, ctx in enumerate(contexts):
                        st.markdown(f"**Context {i+1}:**\n{ctx}")
                        
                # Save to history
                st.session_state.chat_history.extend(
                    [HumanMessage(content=user_query), AIMessage(content=answer)]
                )
                
            except Exception as e:
                st.error(f"Error during generation: {e}")
                st.stop()
                
        # Run Custom Evaluator
        with st.spinner("Evaluating Response..."):
            try:
                eval_result = evaluate_rag_response(
                    question=user_query,
                    answer=answer,
                    contexts=contexts
                )
                
                st.markdown("---")
                st.markdown("**LLM Judge Metrics:**")
                cols = st.columns(len(eval_result))
                for i, (metric_name, score) in enumerate(eval_result.items()):
                    with cols[i]:
                        st.metric(label=metric_name.replace("_", " ").title(), value=f"{score:.2f}")
                        
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
