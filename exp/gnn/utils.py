import torch
import dgl
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from exp.utils import normalize_embeddings


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


def sample_item_batch(user_batch, bipartite_graph):
    sampled_edges = dgl.sampling.sample_neighbors(
        bipartite_graph, {"User": user_batch}, fanout=2
    ).edges(etype="ItemUser")
    item_batch = sampled_edges[0]
    item_batch = item_batch[torch.argsort(sampled_edges[1])]
    item_batch = item_batch.reshape(-1, 2)
    item_batch = item_batch.T
    return item_batch


@torch.no_grad()
def inference_model(model, bipartite_graph, batch_size, hidden_dim, device):
    model.eval()
    item_embeddings = torch.zeros(bipartite_graph.num_nodes("Item"), hidden_dim).to(device)
    for items_batch in tqdm(torch.utils.data.DataLoader(
            torch.arange(bipartite_graph.num_nodes("Item")), 
            batch_size=batch_size, 
            shuffle=True
    )):
        item_embeddings[items_batch] = model(items_batch.to(device))

    item_embeddings = normalize_embeddings(item_embeddings.cpu().numpy())
    return item_embeddings