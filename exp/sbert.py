import argparse
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from utils import normalize_embeddings


def prepare_sbert_embeddings(
        items_path, 
        embeddings_savepath, 
        model_name,
        batch_size,
        device
):
    items = pd.read_csv(items_path).sort_values("item_id")
    sentences = items["description"].values
    model = SentenceTransformer(model_name).to(device)
    model.eval()
    embeddings = []
    for start_index in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[start_index:start_index+batch_size]
        embeddings.extend(model.encode(batch))
    embeddings = normalize_embeddings(np.array(embeddings))
    np.save(embeddings_savepath, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SBERT embeddings.")
    parser.add_argument("--items_path", type=str, required=True, help="Path to the items file.")
    parser.add_argument("--embeddings_savepath", type=str, required=True, help="Path to save the embeddings.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Name of the SBERT model to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda).")
    args = parser.parse_args()

    prepare_sbert_embeddings(**vars(args))