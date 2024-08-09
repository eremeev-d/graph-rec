import numpy as np


def normalize_embeddings(embeddings):
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    nonzero_embeddings = embeddings_norm > 0.0 
    embeddings[nonzero_embeddings] /= embeddings_norm[nonzero_embeddings, None]
    return embeddings