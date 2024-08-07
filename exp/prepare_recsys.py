import argparse
import sqlite3
import io
import os

import faiss
import pandas as pd
import numpy as np


def convert_numpy_array_to_text(array):
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)
    return sqlite3.Binary(stream.read())


def prepare_items_db(items_path, embeddings_path, db_path):
    items = pd.read_csv(items_path)
    embeddings = np.load(embeddings_path)
    items["embedding"] = np.split(embeddings, embeddings.shape[0])

    sqlite3.register_adapter(np.ndarray, convert_numpy_array_to_text)
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        items.to_sql("items", conn, if_exists="replace", index=False, dtype={"embedding": "embedding"})


def build_index(embeddings_path, save_path, n_neighbors):
    embeddings = np.load(embeddings_path)
    index = faiss.IndexHNSWFlat(embeddings.shape[-1], n_neighbors)
    index.add(embeddings)
    faiss.write_index(index, save_path)


def prepare_recsys(
    items_path,
    embeddings_path,
    save_directory,
    n_neighbors=32,
):
    prepare_items_db(items_path, embeddings_path, os.path.join(save_directory, "items.db"))
    build_index(embeddings_path, os.path.join(save_directory, "index.faiss"), n_neighbors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare items database and HNSW index from a CSV file and embeddings.")

    parser.add_argument("--items_path", required=True, type=str, help="Path to the CSV file containing items.")
    parser.add_argument("--embeddings_path", required=True, type=str, help="Path to the .npy file containing item embeddings.")
    parser.add_argument("--save_directory", required=True, type=str, help="Path to the save directory.")
    parser.add_argument("--n_neighbors", type=int, default=32, help="Number of neighbors for the index.")

    args = parser.parse_args()
    prepare_recsys(**vars(args))