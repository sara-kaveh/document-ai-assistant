def build_prompt(question, context, chat_history=""):

    prompt = f"""You are a document question-answering assistant.

Answer the CURRENT QUESTION using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not in the context, say:
   "I couldn't find enough information about this in the document."
3. Answer the question directly.
4. Keep the answer concise.
5. Do not mention source numbers.
6. Do not invent information.

Conversation history:
----------------
{chat_history}
----------------

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    return prompt
