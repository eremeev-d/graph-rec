import argparse

import faiss
import numpy as np


def build_index(embeddings_path, save_path, n_neighbors):
    embeddings = np.load(embeddings_path)
    index = faiss.IndexHNSWFlat(embeddings.shape[-1], 32)
    index.add(embeddings)
    faiss.write_index(index, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an HNSW index from embeddings.")
    parser.add_argument("--embeddings_path", required=True, type=str, help="Path to the embeddings file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the built index.")
    parser.add_argument("--n_neighbors", type=int, default=32, help="Number of neighbors for the index.")
    args = parser.parse_args()
    build_index(**vars(args))