from .prompt import build_prompt
from src.chat_history import ChatHistory


class RAGPipeline:
    """Coordinates query rewriting, retrieval, prompt construction, and generation."""

    def __init__(self, retriever, llm, query_rewriter):
        self.retriever = retriever
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.chat_history = ChatHistory()

    def ask(self, question, k=3, threshold=0.40):
        """Generate an answer using retrieved document context."""

        # Use conversation history to resolve references from previous turns.
        history = self.chat_history.format_history()

        # Rewrite the question into a standalone search query when history exists.
        if history:
            search_query = self.query_rewriter.rewrite(
                question,
                history
            )

        else:
            search_query = question

        print("\n" + "-" * 70)
        print("Search query:", search_query)

        # Retrieve the most relevant document chunks for the search query.
        results = self.retriever.retrieve(
            search_query,
            k=k,
            threshold=threshold
        )

        if not results:
            answer = (
                "I couldn't find enough information "
                "about this in the document."
            )

            self.chat_history.add_user_message(question)
            self.chat_history.add_ai_message(answer)

            return answer, []

        context_parts = []

        for i, result in enumerate(results, start=1):
            chunk = result["chunk"]

            page_number = chunk.metadata.get("page")

            # PyPDFLoader uses zero-based page numbers.
            if page_number is not None:
                page_number += 1

            page_text = (
                page_number
                if page_number is not None
                else "unknown"
            )

            source = chunk.metadata.get("source", "unknown")
            section = chunk.metadata.get("section", "Unknown")

            context_parts.append(
                f"[Source {i} | File: {source} | Page: {page_text}]\n"
                f"Section: {section}\n"
                f"{chunk.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Combine the user's question, retrieved context, and conversation history into the final prompt sent to the language model.
        prompt = build_prompt(
            question=question,
            context=context,
            chat_history=history
        )

        # Generate the final answer using the language model.
        answer = self.llm.generate(prompt)

        # Store the exchange so future questions can use conversation context.
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(answer)

        return answer, results
