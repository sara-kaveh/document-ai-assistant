from src.llm import LLM


class QueryRewriter:

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question, chat_history):

        prompt = f"""Rewrite the user's latest question into a standalone search query.

Conversation:
{chat_history}

Latest question:
{question}

Rules:
1. Replace pronouns and vague references using the conversation.
2. Include the main subject from the conversation.
3. Do not answer the question.
4. Return ONLY the rewritten question.
5. Do not write "Standalone question:".
6. Do not explain anything.

Example:

Conversation:
User: What is the Transformer architecture?
Assistant: The Transformer is an architecture based on attention.

Latest question:
What about the encoder?

Output:
What is the role of the Transformer encoder?

Now do the same.

Conversation:
{chat_history}

Latest question:
{question}

Output:
"""

        rewritten_question = self.llm.generate(prompt)

        return rewritten_question.strip()
