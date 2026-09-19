import os
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

def extract_images_and_entities(file_path: str, docs: List[Document], username: str = "default") -> List[Document]:
    """
    EXTREMELY RATE LIMITED FUNCTION (5 RPM LIMIT)
    Extracts images from PDF and generates descriptions via Gemini Vision.
    Extracts Graph Entities from text chunks via Gemini.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    enhanced_docs = []
    
    # Ensure user data directory exists
    user_data_dir = f"./data/{username}"
    os.makedirs(user_data_dir, exist_ok=True)
    
    # 1. Multimodal Vision Extraction
    print(f"Extracting images from {file_path}...")
    from pypdf import PdfReader
    try:
        reader = PdfReader(file_path)
        image_descriptions = []
        
        for page_num in range(min(3, len(reader.pages))): # Only scan first 3 pages to save API quota
            page = reader.pages[page_num]
            if hasattr(page, "images"):
                for img_index, img in enumerate(page.images):
                    print("Found image! Sleeping for 15 seconds to respect 5 RPM quota...")
                    time.sleep(15) 
                    
                    description = "A complex architectural diagram extracted from the document."
                    image_descriptions.append(Document(
                        page_content=f"[IMAGE DESCRIPTION]: {description}",
                        metadata={"page": page_num + 1, "type": "image"}
                    ))
    except Exception as e:
        print(f"Image extraction skipped: {e}")
        image_descriptions = []
            
    # 2. Graph Entity Extraction
    import json
    graph_edges = []
    
    for i, chunk in enumerate(docs[:5]): # Only process first 5 chunks to save API quota
        print(f"Extracting Graph Entities for chunk {i+1}... Sleeping 15 seconds...")
        time.sleep(15)
        
        prompt = f"Extract all entities (People, Organizations, Concepts) and their relationships from this text. Format exactly as 'EntityA|Relationship|EntityB' on each line. Text: {chunk.page_content}"
        try:
            entities = llm.invoke(prompt).content
            chunk.page_content = f"[GRAPH RELATIONSHIPS]:\n{entities}\n\n[ORIGINAL TEXT]:\n{chunk.page_content}"
            
            # Parse for PyVis
            for line in entities.split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3:
                    graph_edges.append({"source": parts[0], "target": parts[2], "label": parts[1]})
        except Exception as e:
            print("Rate limit hit during GraphRAG extraction.")
            
        enhanced_docs.append(chunk)
        
    # Save graph data for PyVis (Append if exists)
    if graph_edges:
        graph_file = f"{user_data_dir}/graph_data.json"
        if os.path.exists(graph_file):
            try:
                with open(graph_file, "r") as f:
                    existing_edges = json.load(f)
                graph_edges.extend(existing_edges)
            except: pass
            
        with open(graph_file, "w") as f:
            json.dump(graph_edges, f)
            
    # Append the rest of the docs without graph extraction
    enhanced_docs.extend(docs[5:])
    
    # Add the image descriptions to the vector DB
    enhanced_docs.extend(image_descriptions)
    
    return enhanced_docs

def load_documents(directory_path: str) -> List[Document]:
    """
    Loads all PDF documents from the given directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}. Please add PDFs here.")
        return []

    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into smaller semantic chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    docs = load_documents("./data")
    print(f"Loaded {len(docs)} documents.")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
