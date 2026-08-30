![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented_Generation-purple)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-Embeddings-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![Gemma](https://img.shields.io/badge/Gemma_3-1B-8E75B2)
![License](https://img.shields.io/badge/License-MIT-green)

# Document AI Assistant

A lightweight **Retrieval-Augmented Generation (RAG)** application for asking questions about PDF documents.

The system combines **semantic search with FAISS**, **Sentence Transformers embeddings**, and a local **Gemma 3:1B** language model to retrieve relevant document sections and generate concise answers grounded in the uploaded document.

## Features

* Upload PDF documents
* Semantic document retrieval
* Section-aware text chunking
* Sentence Transformers embeddings
* FAISS vector similarity search
* Query rewriting for conversational questions
* Conversation history
* Local Gemma 3:1B LLM
* Source information with file, page, and section
* Prevents answers from outside the provided document
* Question limit per conversation
* Reset conversation
* FastAPI REST API

## Architecture

```text
                    PDF
                     │
                     ▼
              ┌──────────────┐
              │  PDF Loader  │
              └──────┬───────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Section-Aware       │
          │ Text Splitter       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ SentenceTransformer │
          │     Embeddings      │
          └──────────┬──────────┘
                     │
                     ▼
              ┌─────────────┐
              │    FAISS    │
              │ Vector Store│
              └──────┬──────┘
                     │
                     │
User Question ───────┤
                     ▼
          ┌─────────────────────┐
          │   Query Rewriter    │
          │     Gemma 3:1B      │
          └──────────┬──────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Retriever  │
              └──────┬──────┘
                     │
              Relevant Chunks
                     │
                     ▼
          ┌─────────────────────┐
          │    RAG Pipeline     │
          │ Context + History   │
          └──────────┬──────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Gemma 3:1B  │
              └──────┬──────┘
                     │
                     ▼
               Answer + Sources
```

## Tech Stack

| Technology            | Purpose                                 |
| --------------------- | --------------------------------------- |
| Python                | Core language                           |
| FastAPI               | REST API                                |
| LangChain             | Document processing and LLM integration |
| PyPDFLoader           | PDF loading                             |
| Sentence Transformers | Text embeddings                         |
| FAISS                 | Vector similarity search                |
| Ollama                | Local LLM runtime                       |
| Gemma 3:1B            | Local language model                    |
| Pydantic              | Request/response validation             |

## How It Works

### 1. Upload a PDF

The API accepts a PDF through the `/upload` endpoint.

The document is loaded and divided into smaller sections using a section-aware text splitter.

### 2. Generate Embeddings

Each chunk is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

### 3. Build the FAISS Index

The generated embeddings are normalized and stored in a FAISS inner-product index.

This allows the system to efficiently find semantically similar document chunks.

### 4. Rewrite Conversational Questions

When conversation history exists, the latest question is rewritten into a standalone search query.

For example:

```text
User: What is the Transformer?

AI: The Transformer is an architecture based on attention.

User: What about the encoder?
```

The query rewriter converts the second question into something closer to:

```text
What is the role of the Transformer encoder?
```

This makes retrieval more effective for follow-up questions.

### 5. Retrieve Relevant Context

The rewritten question is embedded and compared against the FAISS index.

Only chunks above the similarity threshold are returned.

### 6. Generate the Answer

The retrieved chunks, current question, and conversation history are passed to Gemma 3:1B.

The model is instructed to answer using **only the retrieved document context**.

If the required information cannot be found, the system returns:

```text
I couldn't find enough information about this in the document.
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sara-kaveh/document-ai-assistant.git

cd document-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama and download the model:

```bash
ollama pull gemma3:1b
```

Make sure Ollama is running before starting the application.

### 5. Start the FastAPI server

For example:

```bash
uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### `POST /upload`

Upload a PDF and create a new RAG pipeline.

**Request:**

```text
multipart/form-data
file=<document.pdf>
```

### `POST /ask`

Ask a question about the uploaded document.

**Request:**

```json
{
  "question": "How many encoder layers does the Transformer have?"
}
```

**Response:**

```json
{
  "answer": "The Transformer uses six encoder layers.",
  "sources": [
    {
      "file": "attention.pdf",
      "page": 3,
      "section": "3.1 Encoder and Decoder Stacks"
    }
  ],
  "questions_remaining": 4
}
```

### `POST /reset`

Clears the conversation history and resets the question counter.

## Grounded Answers

The assistant is designed to avoid relying on external knowledge.

The final generation prompt instructs the model to:

* Use only the retrieved context
* Answer the current question directly
* Keep responses concise
* Avoid inventing information
* Return a fallback response when the document does not contain enough information

This helps reduce hallucinations in document-based question answering.

## Current Limitations

* Only PDF documents are supported.
* The vector index is created in memory when a document is uploaded.
* The current implementation processes one active document at a time.
* Conversation state is maintained in memory.
* The application currently limits each conversation to 5 questions.
* Gemma 3:1B is intentionally used as a small local model rather than a large hosted model.

## Purpose

This project was built to explore the implementation of a complete **Retrieval-Augmented Generation pipeline**, from document ingestion and embedding generation to vector retrieval, query rewriting, conversational context, and grounded answer generation.

---

## Project Structure

```text
document-ai-assistant/
│
├── data/
│   └── uploads/
│
├── src/
│   ├── rag/
│   │   ├── prompt.py
│   │   ├── rag_pipeline.py
│   │   └── query_rewriter.py
│   │
│   ├── api.py
│   ├── chat_history.py
│   ├── embedding_generator.py
│   ├── faiss_store.py
│   ├── llm.py
│   ├── main.py
│   ├── pdf_loader.py
│   ├── retriever.py
│   └── text_splitter.py
│
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the MIT License.

---

## Author

**Sara kaveh**

GitHub: https://github.com/sara-kaveh
