import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        faiss.normalize_L2(
            embeddings
        )

        self.index.add(
            embeddings
        )

    def search(self, query_embedding, k=3):

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        return scores, indices
