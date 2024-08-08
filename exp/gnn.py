import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import dgl
import torch
import wandb
from tqdm.auto import tqdm

from exp.utils import prepare_graphs, normalize_embeddings, LRSchedule
from exp.prepare_recsys import prepare_recsys
from exp.evaluate import evaluate_recsys


class GNNLayer(torch.nn.Module):
    def __init__(self, hidden_dim, aggregator_type, skip_connection, bidirectional):
        super().__init__()
        self._skip_connection = skip_connection
        self._bidirectional = bidirectional

        self._conv = dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self._activation = torch.nn.ReLU()

        if bidirectional:
            self._conv_rev = dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type)
            self._activation_rev = torch.nn.ReLU()

    def forward(self, graph, x):
        edge_weights = graph.edata["weights"]

        y = self._activation(self._conv(graph, x, edge_weights))
        if self._bidirectional:
            reversed_graph = dgl.reverse(graph, copy_edata=True)
            edge_weights = reversed_graph.edata["weights"]
            y = y + self._activation_rev(self._conv_rev(reversed_graph, x, edge_weights))

        if self._skip_connection:
            return x + y
        else:
            return y


class GNNModel(torch.nn.Module):
    def __init__(
            self,
            bipartite_graph,
            text_embeddings,
            num_layers,
            hidden_dim,
            aggregator_type,
            skip_connection,
            bidirectional,
            num_traversals, 
            termination_prob, 
            num_random_walks, 
            num_neighbor,
    ):
        super().__init__()

        self._bipartite_graph = bipartite_graph
        self._text_embeddings = text_embeddings

        self._sampler = dgl.sampling.PinSAGESampler(
            bipartite_graph, "Item", "User", num_traversals, 
            termination_prob, num_random_walks, num_neighbor)

        self._text_encoder = torch.nn.Linear(text_embeddings.shape[-1], hidden_dim)

        self._layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self._layers.append(GNNLayer(
                hidden_dim, aggregator_type, skip_connection, bidirectional))

    def _sample_subraph(self, frontier_ids):
        num_layers = len(self._layers)
        device = self._bipartite_graph.device

        subgraph = dgl.graph(([], []), num_nodes=self._bipartite_graph.num_nodes("Item")).to(device)
        prev_ids = set()
        weights = []

        for _ in range(num_layers):
            frontier_ids = torch.tensor(frontier_ids, dtype=torch.int64).to(device)
            new_sample = self._sampler(frontier_ids)
            new_weights = new_sample.edata["weights"]
            new_edges = new_sample.edges()

            subgraph.add_edges(*new_edges)
            weights.append(new_weights)

            prev_ids |= set(frontier_ids.cpu().tolist())
            frontier_ids = set(dgl.compact_graphs(subgraph).ndata[dgl.NID].cpu().tolist())
            frontier_ids = list(frontier_ids - prev_ids)
            
        subgraph.edata["weights"] = torch.cat(weights, dim=0).to(torch.float32)
        return subgraph

    def forward(self, ids):
        ### Sample subgraph
        sampled_subgraph = self._sample_subraph(ids)
        sampled_subgraph = dgl.compact_graphs(sampled_subgraph, always_preserve=ids)

        ### Encode text embeddings
        text_embeddings = self._text_embeddings[
            sampled_subgraph.ndata[dgl.NID]]
        features = self._text_encoder(text_embeddings)

        ### GNN goes brr...
        for layer in self._layers:
            features = layer(sampled_subgraph, features)

        ### Select features for initial ids
        # TODO: write it more efficiently?
        matches = sampled_subgraph.ndata[dgl.NID].unsqueeze(0) == ids.unsqueeze(1)
        ids_in_subgraph = matches.nonzero(as_tuple=True)[1]
        features = features[ids_in_subgraph]
        
        ### Normalize and return
        features = features / torch.linalg.norm(features, dim=1, keepdim=True)
        return features


### Based on https://arxiv.org/pdf/2205.03169
def nt_xent_loss(sim, temperature):
    sim = sim / temperature
    n = sim.shape[0] // 2  # n = |user_batch|

    aligment_loss = -torch.mean(sim[torch.arange(n), torch.arange(n)+n])

    mask = torch.diag(torch.ones(2*n, dtype=torch.bool)).to(sim.device)
    sim = torch.where(mask, -torch.inf, sim)
    sim = sim[:n, :]
    distribution_loss = torch.mean(torch.logsumexp(sim, dim=1))

    loss = aligment_loss + distribution_loss
    return loss


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


def prepare_gnn_embeddings(
        # Paths
        items_path,
        train_ratings_path,
        val_ratings_path,
        text_embeddings_path,
        embeddings_savepath, 
        # Learning hyperparameters
        temperature,
        batch_size, 
        lr, 
        num_epochs, 
        # Model hyperparameters
        num_layers,
        hidden_dim,
        aggregator_type,
        skip_connection,
        bidirectional,
        num_traversals, 
        termination_prob, 
        num_random_walks, 
        num_neighbor,
        # Misc
        validate_every_n_epoch,
        device, 
        wandb_name, 
        use_wandb,
):
    ### Prepare graph
    bipartite_graph, _ = prepare_graphs(items_path, train_ratings_path)
    bipartite_graph = bipartite_graph.to(device)

    ### Init wandb
    if use_wandb:
        wandb.init(project="graph-rec-gnn", name=wandb_name)

    ### Prepare model
    text_embeddings = torch.tensor(np.load(text_embeddings_path)).to(device)
    model = GNNModel(
        bipartite_graph=bipartite_graph, 
        text_embeddings=text_embeddings, 
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        aggregator_type=aggregator_type,
        skip_connection=skip_connection,
        bidirectional=bidirectional,
        num_traversals=num_traversals, 
        termination_prob=termination_prob, 
        num_random_walks=num_random_walks, 
        num_neighbor=num_neighbor
    )
    model = model.to(device)

    ### Prepare dataloader
    all_users = torch.arange(bipartite_graph.num_nodes("User")).to(device)
    all_users = all_users[bipartite_graph.in_degrees(all_users, etype="ItemUser") > 1] # We need to sample 2 items per user
    dataloader = torch.utils.data.DataLoader(
        all_users, batch_size=batch_size, shuffle=True, drop_last=True)

    ### Prepare optimizer & LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    
    ### Train loop
    model.train()
    for epoch in range(num_epochs):
        ### Train
        for user_batch in tqdm(dataloader):
            item_batch = sample_item_batch(user_batch, bipartite_graph)  # (2, |user_batch|)
            item_batch = item_batch.reshape(-1)  # (2 * |user_batch|)
            features = model(item_batch)  # (2 * |user_batch|, hidden_dim)
            sim = features @ features.T  # (2 * |user_batch|, 2 * |user_batch|)
            loss = nt_xent_loss(sim, temperature)
            if use_wandb:
                wandb.log({"loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        ### Validation
        if (validate_every_n_epoch is not None) and (((epoch + 1) % validate_every_n_epoch) == 0):
            item_embeddings = inference_model(
                model, bipartite_graph, batch_size, hidden_dim, device)
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                tmp_embeddings_path = os.path.join(tmp_dir_name, "embeddings.npy")
                np.save(tmp_embeddings_path, item_embeddings)
                prepare_recsys(items_path, tmp_embeddings_path, tmp_dir_name)
                metrics = evaluate_recsys(
                    val_ratings_path, 
                    os.path.join(tmp_dir_name, "index.faiss"),
                    os.path.join(tmp_dir_name, "items.db"))
                print(f"Epoch {epoch + 1} / {num_epochs}. {metrics}")
                if use_wandb:
                    wandb.log(metrics)
            
        
         
    if use_wandb:
        wandb.finish()
    
    ### Process full dataset
    item_embeddings = inference_model(model, bipartite_graph, batch_size, hidden_dim, device)
    np.save(embeddings_savepath, item_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GNN Embeddings")

    # Paths
    parser.add_argument("--items_path", type=str, required=True, help="Path to the items file")
    parser.add_argument("--train_ratings_path", type=str, required=True, help="Path to the train ratings file")
    parser.add_argument("--val_ratings_path", type=str, required=True, help="Path to the validation ratings file")
    parser.add_argument("--text_embeddings_path", type=str, required=True, help="Path to the text embeddings file")
    parser.add_argument("--embeddings_savepath", type=str, required=True, help="Path to the file where gnn embeddings will be saved")

    # Learning hyperparameters
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for NT-Xent loss")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs")

    # Model hyperparameters
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--aggregator_type", type=str, default="mean", help="Type of aggregator in SAGEConv")
    parser.add_argument("--skip_connection", action="store_true", dest="skip_connection", help="Disable skip connections")
    parser.add_argument("--no_bidirectional", action="store_false", dest="bidirectional", help="Do not use reversed edges in convolution")
    parser.add_argument("--num_traversals", type=int, default=4, help="Number of traversals in PinSAGE-like sampler")
    parser.add_argument("--termination_prob", type=float, default=0.5, help="Termination probability in PinSAGE-like sampler")
    parser.add_argument("--num_random_walks", type=int, default=200, help="Number of random walks in PinSAGE-like sampler")
    parser.add_argument("--num_neighbor", type=int, default=10, help="Number of neighbors in PinSAGE-like sampler")

    # Misc
    parser.add_argument("--validate_every_n_epoch", type=int, default=4, help="Perform RecSys validation every n train epochs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--wandb_name", type=str, help="WandB run name")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable WandB logging")

    args = parser.parse_args()

    prepare_gnn_embeddings(**vars(args))