# HackerX 2.0 - Medical AI Chatbot with RAG

<div align="center">

![HackerX 2.0](https://img.shields.io/badge/HackerX%202.0-Medical%20AI%20Chatbot-blue?style=for-the-badge&logo=huggingface&logoColor=white)

**Intelligent Medical Document Q&A System powered by PubMedBERT & BioMistral**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-121212?style=flat-square)](https://langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=flat-square)](https://gradio.app/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-4A90E2?style=flat-square)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/)

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [API Reference](#api-reference)

</div>

---

## Overview

**HackerX 2.0** is a Retrieval-Augmented Generation (RAG) system designed for medical document question-answering. It combines domain-specific embeddings (PubMedBERT) with biomedical language models (BioMistral/Llama 3.2) to provide accurate, context-aware responses from medical literature.

### The Problem

Healthcare professionals and medical students face challenges:
- Quickly finding relevant information in dense medical textbooks
- Getting accurate answers from oncology handbooks and clinical guidelines
- Synthesizing information from multiple medical documents
- Accessing medical knowledge without reading entire documents

### Our Solution

HackerX 2.0 provides:
- **PubMedBERT Embeddings**: Domain-specific embeddings trained on biomedical literature
- **BioMistral/Llama 3.2**: Medical-tuned language models for accurate generation
- **RAG Pipeline**: Retrieves relevant document chunks before generating responses
- **Interactive Chatbot**: Gradio-based interface with chat history and references

## Features

### 1. Medical Document RAG
Retrieval-Augmented Generation optimized for medical content:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEDICAL KNOWLEDGE BASE                             │
├─────────────────────────────────────────────────────────────────────────┤
│ 📚 Medical Oncology Handbook (June 2020 Edition)                        │
│ 📚 Internal Medicine Textbook (Getachew Tizazu, Tadesse Anteneh)        │
│ 📚 Cancer and Cure: A Critical Analysis                                 │
│ 📚 Custom uploaded medical documents (PDF, DOC)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Domain-Specific Embeddings
Using **NeuML/pubmedbert-base-embeddings** trained on:
- PubMed abstracts
- Biomedical literature
- Clinical documentation

### 3. Biomedical Language Models
Configurable LLM backends:
- **BioMistral-7B**: Fine-tuned for biomedical text
- **Llama 3.2-3B-Instruct**: General-purpose with medical prompting

### 4. Interactive Chat Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MED-APP CHATBOT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ 🧑 User: What are the treatment options for stage 2 breast cancer?│ │
│  │                                                                    │ │
│  │ 🤖 Assistant: Based on the medical oncology handbook, stage 2     │ │
│  │    breast cancer treatment typically includes:                    │ │
│  │    1. Surgery (lumpectomy or mastectomy)                          │ │
│  │    2. Radiation therapy                                           │ │
│  │    3. Chemotherapy (adjuvant or neoadjuvant)                      │ │
│  │    4. Hormone therapy if ER/PR positive...                        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Enter your medical question...                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  [Submit] [📁 Upload PDF] [References] [Clear]                         │
│                                                                         │
│  Temperature: [0.1 ━━●━━━━━━━━━ 1.0]                                   │
│  Top-K:       [10  ━━●━━━━━━━━━ 100]                                   │
│  Top-P:       [0.1 ━━●━━━━━━━━━ 1.0]                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5. Reference Tracking
View source documents and page numbers for all retrieved content:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📄 REFERENCES                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  # Retrieved content 1:                                                 │
│  "Breast cancer staging follows the TNM system where T describes..."    │
│  Source: medical_oncology_handbook.pdf | Page: 127 | [View PDF]         │
│                                                                         │
│  # Retrieved content 2:                                                 │
│  "Treatment protocols for stage II breast cancer include..."            │
│  Source: cancer_and_cure.pdf | Page: 45 | [View PDF]                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GRADIO CHATBOT (Port 7860)                        │   │
│  │   • Chat history with like/dislike feedback                         │   │
│  │   • PDF/DOC upload for custom documents                             │   │
│  │   • Adjustable temperature, top_k, top_p                            │   │
│  │   • Reference sidebar with source links                             │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE                                       │
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   User      │    │    PubMedBERT   │    │        ChromaDB             │ │
│  │   Query     │───▶│    Embeddings   │───▶│     Vector Search           │ │
│  │             │    │                 │    │     (k=2 chunks)            │ │
│  └─────────────┘    └─────────────────┘    └──────────────┬──────────────┘ │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PROMPT CONSTRUCTION                               │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ System: Answer based on given content only                  │    │   │
│  │  │ Chat History: [Previous Q&A pairs]                          │    │   │
│  │  │ Retrieved Content 1: [chunk from oncology handbook]         │    │   │
│  │  │ Retrieved Content 2: [chunk from internal medicine]         │    │   │
│  │  │ User Question: [current query]                              │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LLM SERVICE (Port 8888)                         │   │
│  │   Model: BioMistral-7B / Llama-3.2-3B-Instruct                      │   │
│  │   Config: temperature=0.1, top_k=10, top_p=0.1                      │   │
│  │   Max tokens: 8192                                                   │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    REFERENCE SERVER (Port 8000)                      │   │
│  │   Serves PDF documents for inline viewing                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE BASE                                       │
│  ┌──────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  data/docs/                  │  │  data/vectordb/processed/chroma/   │  │
│  │  ├── medical_oncology.pdf    │  │  └── chroma.sqlite3                │  │
│  │  ├── internal_medicine.pdf   │  │      (PubMedBERT embeddings)       │  │
│  │  └── cancer_and_cure.pdf     │  │                                    │  │
│  └──────────────────────────────┘  └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.x (recommended for GPU acceleration)
- 16GB+ RAM (for BioMistral-7B)
- HuggingFace account with model access

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Sri-Karthik-Avala/hackerx2.0.git
cd hackerx2.0
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install langchain langchain-community chromadb
pip install transformers accelerate bitsandbytes
pip install gradio flask pyprojroot python-dotenv pyyaml
pip install sentence-transformers pypdf
```

4. **Configure environment**
```bash
# Create .env file
echo "GEMMA_TOKEN=your_huggingface_token" > .env
```

5. **Prepare the vector database**
```bash
python src/upload_data_manually.py
```

6. **Start the services**
```bash
# Option 1: All-in-one (multi-threaded)
python main.py

# Option 2: Individual services
python src/llm_serve.py      # LLM API on port 8888
python src/app.py            # Gradio app on port 7860
```

## Usage

### Running the Medical Chatbot

1. Start the main server:
```bash
python main.py
```

2. Open Gradio interface at the provided URL (or http://localhost:7860)

3. Choose data source:
   - **Preprocessed doc**: Use pre-indexed medical documents
   - **Upload doc**: Upload your own PDFs for RAG

4. Ask medical questions:
   - "What are the side effects of chemotherapy?"
   - "Explain the staging system for lung cancer"
   - "What is the recommended treatment for hypertension?"

5. Adjust generation parameters:
   - **Temperature**: Lower = more focused, Higher = more creative
   - **Top-K**: Number of tokens to consider at each step
   - **Top-P**: Cumulative probability threshold

### Using the API Directly

```python
import requests

# Generate medical response
response = requests.post(
    "http://localhost:8888/generate_text",
    json={
        "prompt": [{"role": "user", "content": "What is metastasis?"}],
        "max_new_tokens": 500,
        "temperature": 0.1,
        "top_k": 10,
        "top_p": 0.1
    }
)
print(response.json()["response"])
```

## Project Structure

```
hackerx2.0/
├── main.py                      # Multi-threaded combined server
├── configs/
│   └── app_config.yml           # LLM, retrieval, and server config
├── src/
│   ├── app.py                   # Gradio medical chatbot interface
│   ├── llm_serve.py             # Flask LLM API server
│   ├── llm_service.py           # LLM service implementation
│   ├── reference_serve.py       # PDF document server
│   ├── upload_data_manually.py  # Vector DB preparation script
│   └── utils/
│       ├── chatbot.py           # RAG chatbot logic
│       ├── prepare_vectordb.py  # ChromaDB preparation
│       ├── load_config.py       # YAML configuration loader
│       ├── ui_settings.py       # Gradio UI utilities
│       └── upload_file.py       # File upload handling
├── utils/                       # Duplicate utilities
├── data/
│   ├── docs/                    # Medical PDF documents
│   │   ├── medical_oncology_handbook.pdf
│   │   ├── internal_medicine.pdf
│   │   └── cancer_and_cure.pdf
│   └── vectordb/
│       └── processed/chroma/    # ChromaDB storage
├── images/
│   ├── test.png                 # User avatar
│   └── Gemma-logo.png           # Bot avatar
└── README.md
```

## Configuration

### app_config.yml

```yaml
llm_config:
  engine: "BioMistral/BioMistral-7B"          # Or Llama-3.2-3B-Instruct
  embedding_model: "NeuML/pubmedbert-base-embeddings"
  device: "cuda"                               # Or "cpu"
  temperature: 0.1
  top_k: 10
  top_p: 0.1
  max_new_tokens: 8192
  add_history: false                           # Include chat history in prompt

splitter_config:
  chunk_size: 1500                             # Characters per chunk
  chunk_overlap: 250                           # Overlap between chunks

retrieval_config:
  k: 2                                         # Number of chunks to retrieve

memory:
  number_of_q_a_pairs: 2                       # Chat history length
```

## API Reference

### LLM Generation Endpoint

**POST** `/generate_text` (Port 8888)

```json
{
  "prompt": [{"role": "user", "content": "Your medical question"}],
  "max_new_tokens": 1000,
  "temperature": 0.1,
  "top_k": 10,
  "top_p": 0.1,
  "do_sample": true
}
```

**Response:**
```json
{
  "response": "Based on medical literature, the answer is..."
}
```

## Technologies Used

| Component | Technology |
|-----------|------------|
| **Embeddings** | NeuML/pubmedbert-base-embeddings |
| **LLM** | BioMistral-7B, Llama-3.2-3B-Instruct |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain |
| **Web UI** | Gradio |
| **API Server** | Flask |
| **Document Processing** | PyPDF, LangChain Loaders |

## Future Enhancements

1. **Meditron Integration**: Add Meditron-70B for enhanced medical reasoning
2. **Multi-Modal Support**: Process medical images alongside text
3. **Citation Generation**: Automatic medical citation formatting
4. **Clinical Guidelines**: Integrate WHO/CDC guidelines
5. **Drug Interaction Checker**: Cross-reference medications
6. **HIPAA Compliance**: Secure deployment for clinical use

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/medical-feature`)
3. Commit your changes (`git commit -m 'Add medical feature'`)
4. Push to the branch (`git push origin feature/medical-feature`)
5. Open a Pull Request

## Disclaimer

This system is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

- **Avala Sri Karthik** - *Lead Developer*
- **Charan** - *Collaborator*

**Project**: HackerX 2.0 - Hackathon Project
**Year**: 2024-2025

For questions or collaborations, feel free to reach out!
