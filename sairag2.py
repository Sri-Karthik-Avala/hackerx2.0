import streamlit as st
import os
import sys
import hashlib
import torch
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, JSONLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv(override=True)

# Initialize Streamlit UI
st.markdown("<h1 style='text-align: center;'>Local RAG Chatbot v2</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_embeddings():
    """Load sentence transformer embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_llm():
    """Load LLaMA 3.2 model and tokenizer."""
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

@st.cache_data
def process_document(file_path, file_type):
    """Loads and processes a document based on its type."""
    loader_map = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "json": JSONLoader, 
        "xlsx": UnstructuredExcelLoader,
        "pptx": UnstructuredPowerPointLoader
    }
    if file_type not in loader_map:
        return []

    loader = loader_map[file_type](file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def get_file_hash(content):
    """Generates a unique hash for the uploaded file."""
    return hashlib.sha256(content).hexdigest()

# File Upload
uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "xlsx", "pptx", "json"])

if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    file_hash = get_file_hash(file_content)
    file_extension = uploaded_file.name.split(".")[-1]

    # Define temp directory
    base_dir = os.path.join(tempfile.gettempdir(), "rag_chatbot_v2")
    doc_path = os.path.join(base_dir, file_hash, f"doc.{file_extension}")
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    # Save document
    if not os.path.exists(doc_path):
        with open(doc_path, "wb") as f:
            f.write(file_content)

    # Create vector database directory
    if "chroma_dir" not in st.session_state:
        st.session_state.chroma_dir = os.path.join(base_dir, file_hash, "chroma")

    # Process document and store embeddings if not already stored
    if not os.path.exists(st.session_state.chroma_dir):
        with st.status("📤 Processing document...", expanded=True):
            docs = process_document(doc_path, file_extension)
            embeddings = load_embeddings()
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=st.session_state.chroma_dir
            )

    # Load vector database
    embeddings = load_embeddings()
    db = Chroma(persist_directory=st.session_state.chroma_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # Load LLM
    model, tokenizer = load_llm()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask about the document:")

    if query:
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Format prompt for LLM
        input_text = (
            f"You are an AI assistant answering queries based on retrieved documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)

        raw_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean response: Extract answer after "Answer:"
        answer_start = raw_answer.lower().find("answer:")
        answer = raw_answer[answer_start + 7:].strip() if answer_start != -1 else raw_answer

        # Save chat history
        st.session_state.chat_history.append({"user": query, "assistant": answer, "sources": retrieved_docs})

    # Display chat history
    for exchange in st.session_state.chat_history:
        st.chat_message("user").write(exchange["user"])
        st.chat_message("assistant").write(exchange["assistant"])

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
3. Your data remains locally stored in a temp directory.
""")
