import streamlit as st
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import hashlib
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables
load_dotenv(override=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_data
def process_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == "xlsx":
        loader = UnstructuredExcelLoader(file_path)
    elif file_type == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_type == "json":
        loader = JSONLoader(file_path)
    else:
        return []
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)

@st.cache_resource
def load_llm():
    return AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment="gpt-4o",
        api_version="2023-09-01-preview",
        temperature=0.6,
        streaming=True
    )

st.markdown("<h1 style='text-align: center;'>Local RAG Chatbot</h1>", unsafe_allow_html=True)

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def get_file_hash(content):
    return hashlib.sha256(content).hexdigest()

uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "xlsx", "pptx", "json"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[-1]
    file_content = uploaded_file.getvalue()
    file_hash = get_file_hash(file_content)
    base_dir = os.path.join(tempfile.gettempdir(), "rag_chatbot")
    
    if "chroma_dir" not in st.session_state:
        st.session_state.chroma_dir = os.path.join(base_dir, file_hash, "chroma")
    
    doc_path = os.path.join(base_dir, file_hash, f"doc.{file_type}")
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    
    if not os.path.exists(doc_path):
        with open(doc_path, "wb") as f:
            f.write(file_content)
    
    if not os.path.exists(st.session_state.chroma_dir):
        with st.status("📤 Processing document...", expanded=True):
            docs = process_document(doc_path, file_type)
            embeddings = load_embeddings()
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=st.session_state.chroma_dir
            )
    
    embeddings = load_embeddings()
    db = Chroma(
        persist_directory=st.session_state.chroma_dir,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={'k': 3})
    
    llm = load_llm()
    prompt = ChatPromptTemplate.from_template("""
    Answer using only this context:
    {context}
    Question: {input}
    """)
    
    retrieval_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )
    
    query = st.chat_input("Ask about the document:")
    if query:
        try:
            with st.status("🔍 Processing...", expanded=False):
                response = retrieval_chain.invoke({"input": query})
            
            answer = response["answer"]
            st.chat_message("assistant").write(answer)
            
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            with st.expander("📚 See sources"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', '?')})")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")

st.sidebar.markdown("""
**How it works:**
1. Upload a document (PDF, DOCX, Excel, PPTX, JSON)
2. The system will:
   - Extract text content
   - Create local vector database
   - Prepare the AI model
3. Ask questions about the document
4. All data persists in a temp directory until system cleanup
""")
