import streamlit as st
import os
import hashlib
import torch
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    Docx2txtLoader, JSONLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    FileSystemBlobLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser

# Load environment variables
load_dotenv(override=True)

# Set up database folder
DB_DIR = Path(r"C:\Users\srika\OneDrive\Desktop\Deskop\New folder\hackerx2.0\data\docs")
DB_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR = DB_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Streamlit UI
st.title("Local RAG Chatbot v2")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_llm():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    except Exception as e:
        st.warning(f"GPU not available, switching to CPU. Error: {e}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    
    return model, tokenizer

@st.cache_data
def process_document(file_path, file_type):
    """Process uploaded documents and extract text."""
    try:
        if file_type == "pdf":
            loader = GenericLoader(
                blob_loader=FileSystemBlobLoader(path=str(file_path.parent), glob=file_path.name),
                blob_parser=PyMuPDFParser(),
            )
            docs = loader.load()
        else:
            loader_map = {
                "docx": Docx2txtLoader,
                "json": JSONLoader,
                "xlsx": UnstructuredExcelLoader,
                "pptx": UnstructuredPowerPointLoader
            }
            loader = loader_map.get(file_type, lambda x: [])(str(file_path))
            docs = loader.load()
        
        if not docs:
            st.error(f"⚠️ No text extracted from {file_path.name}. It might be empty or not supported.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
        return text_splitter.split_documents(docs)
    
    except Exception as e:
        st.error(f"❌ Error processing {file_path.name}: {e}")
        return []

def get_file_hash(content):
    return hashlib.sha256(content).hexdigest()

# Sidebar - Upload and Process Documents
st.sidebar.header("📂 Update Database")
new_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "xlsx", "pptx", "json"])

db_files = list(DB_DIR.glob("*.*"))
if db_files:
    st.sidebar.subheader("Existing Documents:")
    for doc in db_files:
        st.sidebar.text(doc.name)

if new_file:
    file_content = new_file.getvalue()
    file_hash = get_file_hash(file_content)
    file_extension = new_file.name.split(".")[-1]
    doc_path = DB_DIR / f"{file_hash}.{file_extension}"
    
    if not doc_path.exists():
        with open(doc_path, "wb") as f:
            f.write(file_content)
        
        with st.sidebar.status("🔄 Indexing document..."):
            docs = process_document(doc_path, file_extension)
            if docs:
                Chroma.from_documents(docs, embedding=load_embeddings(), persist_directory=str(CHROMA_DIR / file_hash))
                st.sidebar.success("✅ Database Updated!")
            else:
                st.sidebar.error("⚠️ Document could not be indexed.")

# Process existing documents
embeddings = load_embeddings()
retrievers = []
for doc in db_files:
    file_hash = doc.stem
    if not (CHROMA_DIR / file_hash).exists():
        docs = process_document(doc, doc.suffix[1:])
        if docs:
            Chroma.from_documents(docs, embedding=embeddings, persist_directory=str(CHROMA_DIR / file_hash))
    db = Chroma(persist_directory=str(CHROMA_DIR / file_hash), embedding_function=embeddings)
    retrievers.append(db.as_retriever(search_kwargs={'k': 3}))

# Load LLM
model, tokenizer = load_llm()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a question based on the documents:")

if query:
    retrieved_docs = []
    for retriever in retrievers:
        retrieved_docs.extend(retriever.get_relevant_documents(query))
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    if context:
        input_text = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        
        raw_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        answer_start = raw_answer.lower().find("answer:")
        answer = raw_answer[answer_start + 7:].strip() if answer_start != -1 else raw_answer
    else:
        answer = "I couldn't find relevant information in the documents."

    st.session_state.chat_history.append({"user": query, "assistant": answer, "sources": retrieved_docs})

# Display chat history
for exchange in st.session_state.chat_history:
    st.chat_message("user").write(exchange["user"])
    st.chat_message("assistant").write(exchange["assistant"])
    
    if exchange["sources"]:
        with st.expander("📚 See sources"):
            for i, doc in enumerate(exchange["sources"]):
                st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', '?')})")
                st.text(doc.page_content[:300] + "...")
                st.divider()

st.sidebar.markdown("""
### How it works:
1. Upload a document (`PDF, DOCX, XLSX, PPTX, JSON`).
2. The system will:
   - Extract text content.
   - Create a local vector database.
   - Use Llama 3.2 to answer queries.
3. Your data remains locally stored in `db/`.
""")
