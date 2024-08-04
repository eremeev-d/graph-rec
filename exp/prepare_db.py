import argparse
import sqlite3
import io

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare items database from a CSV file.")
    parser.add_argument("--items_path", required=True, type=str, help="Path to the CSV file containing items.")
    parser.add_argument("--embeddings_path", required=True, type=str, help="Path to the .npy file containing item embeddings.")
    parser.add_argument("--db_path", required=True, type=str, help="Path to the SQLite database file.")

    args = parser.parse_args()
    prepare_items_db(**vars(args))