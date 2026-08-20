class Retriever:

    def __init__(self, vector_store, embedder, chunks):

        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks

    def retrieve(self, question, k=3, threshold=0.40):

        # Convert the question to an embedding
        question_embedding = self.embedder.embed_texts(
            [question]
        )[0]

        # Search the vector store for the most similar chunks
        scores, indices = self.vector_store.search(
            question_embedding,
            k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            score = float(score)

            if score >= threshold:

                results.append({
                    "chunk": self.chunks[idx],
                    "score": score
                })

        return results
