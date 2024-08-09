import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import dgl
import torch
import wandb
from tqdm.auto import tqdm

from exp.utils import normalize_embeddings
from exp.prepare_recsys import prepare_recsys
from exp.evaluate import evaluate_recsys
from exp.gnn.model import GNNModel
from exp.gnn.loss import nt_xent_loss
from exp.gnn.utils import (
    prepare_graphs, LRSchedule, 
    sample_item_batch, inference_model)


def prepare_gnn_embeddings(config):
    ### Prepare graph
    bipartite_graph, _ = prepare_graphs(config["items_path"], config["train_ratings_path"])
    bipartite_graph = bipartite_graph.to(config["device"])

    ### Init wandb
    if config["use_wandb"]:
        wandb.init(project="graph-rec-gnn", name=config["wandb_name"], config=config)

    ### Prepare model
    text_embeddings = torch.tensor(np.load(config["text_embeddings_path"])).to(config["device"])
    model = GNNModel(
        bipartite_graph=bipartite_graph, 
        text_embeddings=text_embeddings, 
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        aggregator_type=config["aggregator_type"],
        skip_connection=config["skip_connection"],
        bidirectional=config["bidirectional"],
        num_traversals=config["num_traversals"], 
        termination_prob=config["termination_prob"], 
        num_random_walks=config["num_random_walks"], 
        num_neighbor=config["num_neighbor"]
    )
    model = model.to(config["device"])

    ### Prepare dataloader
    all_users = torch.arange(bipartite_graph.num_nodes("User")).to(config["device"])
    all_users = all_users[bipartite_graph.in_degrees(all_users, etype="ItemUser") > 1] # We need to sample 2 items per user
    dataloader = torch.utils.data.DataLoader(
        all_users, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    ### Prepare optimizer & LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    
    ### Train loop
    for epoch in range(config["num_epochs"]):
        ### Train
        model.train()
        for user_batch in tqdm(dataloader):
            item_batch = sample_item_batch(user_batch, bipartite_graph)  # (2, |user_batch|)
            item_batch = item_batch.reshape(-1)  # (2 * |user_batch|)
            features = model(item_batch)  # (2 * |user_batch|, hidden_dim)
            sim = features @ features.T  # (2 * |user_batch|, 2 * |user_batch|)
            loss = nt_xent_loss(sim, config["temperature"])
            if config["use_wandb"]:
                wandb.log({"loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        ### Validation
        if (config["validate_every_n_epoch"] is not None) and (((epoch + 1) % config["validate_every_n_epoch"]) == 0):
            item_embeddings = inference_model(
                model, bipartite_graph, config["batch_size"], config["hidden_dim"], config["device"])
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                tmp_embeddings_path = os.path.join(tmp_dir_name, "embeddings.npy")
                np.save(tmp_embeddings_path, item_embeddings)
                prepare_recsys(config["items_path"], tmp_embeddings_path, tmp_dir_name)
                metrics = evaluate_recsys(
                    config["val_ratings_path"], 
                    os.path.join(tmp_dir_name, "index.faiss"),
                    os.path.join(tmp_dir_name, "items.db"))
                print(f"Epoch {epoch + 1} / {config['num_epochs']}. {metrics}")
                if config["use_wandb"]:
                    wandb.log(metrics)
         
    if config["use_wandb"]:
        wandb.finish()
    
    ### Process full dataset
    item_embeddings = inference_model(model, bipartite_graph, config["batch_size"], config["hidden_dim"], config["device"])
    np.save(config["embeddings_savepath"], item_embeddings)

    ### Save final model
    torch.save(model.to("cpu").state_dict(), config["model_savepath"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GNN Embeddings")

    # Paths
    parser.add_argument("--items_path", type=str, required=True, help="Path to the items file")
    parser.add_argument("--train_ratings_path", type=str, required=True, help="Path to the train ratings file")
    parser.add_argument("--val_ratings_path", type=str, required=True, help="Path to the validation ratings file")
    parser.add_argument("--text_embeddings_path", type=str, required=True, help="Path to the text embeddings file")
    parser.add_argument("--embeddings_savepath", type=str, required=True, help="Path to the file where gnn embeddings will be saved")
    parser.add_argument("--model_savepath", type=str, required=True, help="Path to save final model checkpoint.")

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

    prepare_gnn_embeddings(vars(args))