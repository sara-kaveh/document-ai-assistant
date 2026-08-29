import os
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from .pdf_loader import load_pdf
from .text_splitter import SectionAwareSplitter
from .embedding_generator import EmbeddingGenerator
from .faiss_store import FAISSVectorStore
from .retriever import Retriever
from .llm import LLM

from src.rag.rag_pipeline import RAGPipeline
from src.rag.query_rewriter import QueryRewriter


app = FastAPI(
    title="Document AI Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)


MAX_QUESTIONS = 5
UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: int | None
    section: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]
    questions_remaining: int


rag = None
question_count = 0
current_file = None


def create_rag_pipeline(pdf_path: str):
    """Build a complete RAG pipeline from an uploaded PDF."""

    # Load the PDF and convert it into document objects.
    documents = load_pdf(pdf_path)

    # Split the document into overlapping, section-aware chunks.
    splitter = SectionAwareSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # Generate embeddings for all document chunks.
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_texts(texts)

    # Build the FAISS vector index for semantic search.
    vector_store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_embeddings(embeddings)

    # Connect the vector store with the original document chunks.
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        chunks=chunks
    )

    # Initialize the language model and query rewriter.
    llm = LLM()
    query_rewriter = QueryRewriter(llm)

    # Assemble the complete RAG pipeline.
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        query_rewriter=query_rewriter
    )


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """Upload a PDF and initialize the RAG pipeline."""

    global rag
    global question_count
    global current_file

    # Validate the uploaded file type.
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:
        # Save the uploaded PDF to the local upload directory.
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        await file.close()

        # Process the PDF and create a new RAG pipeline.
        rag = create_rag_pipeline(file_path)

        # Start a fresh conversation for the new document.
        question_count = 0
        current_file = file.filename

        return {
            "message": "PDF uploaded successfully.",
            "file": file.filename,
            "questions_remaining": MAX_QUESTIONS
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(
    request: QuestionRequest
):
    """Answer a question using the currently uploaded document."""

    global question_count

    if rag is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    if question_count >= MAX_QUESTIONS:
        raise HTTPException(
            status_code=429,
            detail="Question limit reached. Please start a new chat."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Retrieve relevant chunks and generate an answer.
    answer, results = rag.ask(
        request.question,
        k=3
    )

    question_count += 1

    sources = []

    for result in results:
        chunk = result["chunk"]

        page = chunk.metadata.get("page")

        # Convert the zero-based page index to a human-readable page number.
        if page is not None:
            page += 1

        sources.append({
            "file": chunk.metadata.get(
                "source",
                current_file or "Unknown"
            ),
            "page": page,
            "section": chunk.metadata.get(
                "section",
                "Unknown"
            )
        })

    return {
        "answer": answer,
        "sources": sources,
        "questions_remaining": MAX_QUESTIONS - question_count
    }


@app.post("/reset")
def reset_chat():
    """Clear the conversation history and reset the question limit."""

    global question_count

    if rag is not None:
        rag.chat_history.clear()

    question_count = 0

    return {
        "message": "Conversation reset.",
        "questions_remaining": MAX_QUESTIONS
    }
