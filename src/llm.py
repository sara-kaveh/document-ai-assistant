from langchain_ollama import ChatOllama


class LLM:

    def __init__(self):

        self.model = ChatOllama(
            model="gemma3:1b",
            temperature=0,
            num_predict=300
        )

    def generate(self, prompt):

        response = self.model.invoke(prompt)

        return response.content
