import numpy as np
import pandas as pd
import dgl
import torch


def normalize_embeddings(embeddings):
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    nonzero_embeddings = embeddings_norm > 0.0 
    embeddings[nonzero_embeddings] /= embeddings_norm[nonzero_embeddings, None]
    return embeddings


def prepare_graphs(items_path, ratings_path):
    items = pd.read_csv(items_path)
    ratings = pd.read_csv(ratings_path)

    n_users = np.max(ratings["user_id"].unique()) + 1
    item_ids = torch.tensor(sorted(items["item_id"].unique()))

    edges = torch.tensor(ratings["user_id"]), torch.tensor(ratings["item_id"])
    reverse_edges = (edges[1], edges[0])

    bipartite_graph = dgl.heterograph(
        data_dict={
            ("User", "UserItem", "Item"): edges,
            ("Item", "ItemUser", "User"): reverse_edges
        },
        num_nodes_dict={
            "User": n_users,
            "Item": len(item_ids)
        }
    )
    graph = dgl.to_homogeneous(bipartite_graph)
    graph = dgl.add_self_loop(graph)
    return bipartite_graph, graph


def extract_item_embeddings(node_embeddings, bipartite_graph, graph):
    item_ntype = bipartite_graph.ntypes.index("Item")
    item_mask = graph.ndata[dgl.NTYPE] == item_ntype
    item_embeddings = node_embeddings[item_mask]
    original_ids = graph.ndata[dgl.NID][item_mask]
    item_embeddings = item_embeddings[torch.argsort(original_ids)]
    return item_embeddings.cpu().numpy()


class LRSchedule:
    def __init__(self, total_steps, warmup_steps, final_factor):
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps
        self._final_factor = final_factor
        
    def __call__(self, step):
        if step >= self._total_steps:
            return self._final_factor
        
        if self._warmup_steps > 0:
            warmup_factor = step / self._warmup_steps
        else:
            warmup_factor = 1.0
        
        steps_after_warmup = step - self._warmup_steps
        total_steps_after_warmup = self._total_steps - self._warmup_steps
        after_warmup_factor = 1 \
            - (1 - self._final_factor) * (steps_after_warmup / total_steps_after_warmup)
        
        factor = min(warmup_factor, after_warmup_factor)
        return min(max(factor, 0), 1)