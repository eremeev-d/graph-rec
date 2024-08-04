import argparse
import os

import numpy as np
import pandas as pd
import dgl
import torch
import wandb
from tqdm.auto import tqdm

from utils import prepare_graphs, extract_item_embeddings, normalize_embeddings


def prepare_deepwalk_embeddings(
        items_path,
        ratings_path, 
        embeddings_savepath,
        emb_dim, 
        window_size, 
        batch_size, 
        lr, 
        num_epochs, 
        device, 
        wandb_name, 
        use_wandb
):
    ### Prepare graph
    bipartite_graph, graph = prepare_graphs(items_path, ratings_path)
    bipartite_graph = bipartite_graph.to(device)
    graph = graph.to(device)

    ### Run deepwalk
    if use_wandb:
        wandb.init(project="graph-recs-deepwalk", name=wandb_name)

    model = dgl.nn.DeepWalk(graph, emb_dim=emb_dim, window_size=window_size)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.arange(graph.num_nodes()), 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=model.sample)
    
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for batch_walk in tqdm(dataloader):
            loss = model(batch_walk)
            if use_wandb:
                wandb.log({"loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
    if use_wandb:
        wandb.finish()
    
    node_embeddings = model.node_embed.weight.detach().to(device)

    ### Extract & save item embeddings
    item_embeddings = extract_item_embeddings(node_embeddings, bipartite_graph, graph)
    item_embeddings = normalize_embeddings(item_embeddings)
    np.save(embeddings_savepath, item_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DeepWalk embeddings.")
    parser.add_argument("--items_path", type=str, required=True, help="Path to the items file.")
    parser.add_argument("--ratings_path", type=str, required=True, help="Path to the ratings file.")
    parser.add_argument("--embeddings_savepath", type=str, required=True, help="Path to the file where embeddings will be saved.")
    parser.add_argument("--emb_dim", type=int, default=384, help="Dimensionality of the embeddings.")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for the DeepWalk algorithm.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda).")
    parser.add_argument("--wandb_name", type=str, help="Name for WandB run.")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable WandB logging")
    args = parser.parse_args()

    prepare_deepwalk_embeddings(**vars(args))