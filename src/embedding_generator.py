from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
