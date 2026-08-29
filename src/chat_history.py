class ChatHistory:

    def __init__(self):

        self.messages = []

    def add_user_message(self, message):

        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_ai_message(self, message):

        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def format_history(self):

        formatted = []

        for message in self.messages:

            role = message["role"].capitalize()

            content = message["content"]

            formatted.append(
                f"{role}: {content}"
            )

        return "\n\n".join(formatted)

    def clear(self):

        self.messages = []
