import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pdf_loader import load_pdf
from src.text_splitter import SectionAwareSplitter
from src.embedding_generator import EmbeddingGenerator
from src.faiss_store import FAISSVectorStore
from src.retriever import Retriever
from src.llm import LLM
from src.rag.rag_pipeline import RAGPipeline
from src.rag.query_rewriter import QueryRewriter


PDF_PATH = "data/attention.pdf"


def main():
    """Run the end-to-end RAG pipeline on a PDF document."""

    # 1. Load the source PDF and convert it into LangChain documents.
    documents = load_pdf(PDF_PATH)

    # 2. Split the document into overlapping, section-aware chunks.
    splitter = SectionAwareSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # 3. Generate vector embeddings for each document chunk.
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_texts(texts)

    # 4. Build the FAISS vector index from the generated embeddings.
    vector_store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_embeddings(embeddings)

    # 5. Create the retriever that connects semantic search with the original document chunks.
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        chunks=chunks
    )

    # 6. Initialize the language model used for generation.
    llm = LLM()

    # 7. Initialize the query rewriter for conversational questions.
    query_rewriter = QueryRewriter(llm)

    # 8. Assemble the complete RAG pipeline.
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
        query_rewriter=query_rewriter
    )

    # 9. Test factual, conversational, and out-of-document questions.
    questions = [
        "How many encoder layers does the Transformer have?",
        "What is the purpose of multi-head attention?",
        "What does it do?",
        "What is the capital of Germany?",
    ]

    for question in questions:
        answer, results = rag.ask(
            question,
            k=3
        )

        print_result(
            question,
            answer,
            results
        )


def print_result(question, answer, results):
    """Print the question, generated answer, and retrieved sources."""

    print("User:", question)
    print("AI:", answer)

    if not results:
        print("\nNo relevant sources found.")
        return

    print("-" * 70)
    print("SOURCES")

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]
        score = result["score"]

        source = chunk.metadata.get(
            "source",
            "Unknown"
        )

        page = chunk.metadata.get(
            "page",
            None
        )

        # Convert the zero-based page index to a human-readable page number.
        if page is not None:
            page += 1

        print(f"\n[Source {i}]")
        print(f"File: {source}")

        if page is not None:
            print(f"Page: {page}")

        print(f"Similarity score: {score:.4f}")

        print("\nRelevant text:")
        print(chunk.page_content[:500])


if __name__ == "__main__":
    main()
