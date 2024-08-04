import itertools
import os

import numpy as np
import faiss

from app.database import ItemDatabase


class RecommenderSystem:
    def __init__(self, faiss_index_path, db_path):
        self._index = faiss.read_index(faiss_index_path)
        self._db = ItemDatabase(db_path)

    def recommend_items(self, query, n_items=10):
        query_embedding = self._db.get_item(query)["embedding"]
        _, results = self._index.search(query_embedding, k=n_items+1)
        results = filter(lambda item: item != query, results[0])
        return itertools.islice(results, n_items)
