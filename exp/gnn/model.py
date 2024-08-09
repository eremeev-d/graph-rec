import torch
import dgl


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