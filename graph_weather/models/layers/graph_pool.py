import torch
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
import torch.nn.functional as F
import math


class Graph2Vec(torch.nn.Module):
    """Convert output graph to a single vector for attention with the denser model"""

    def __init__(
        self,
        num_input_processor_nodes: int,
        num_output_processor_nodes: int,
        processor_node_dim: int,
        num_layers: int = 3,
        latent_dim: int = 1000
    ):
        super().__init__()

        self.num_output_processor_nodes = num_output_processor_nodes
        self.processor_node_dim = processor_node_dim

        self.layer1 = torch.nn.Linear(in_features=num_input_processor_nodes*processor_node_dim, out_features=latent_dim)
        self.inner_layers = torch.nn.ModuleList([torch.nn.Linear(latent_dim, latent_dim) for _ in range(num_layers-2)])
        self.final_layer = torch.nn.Linear(latent_dim, processor_node_dim)
    

    def forward(self, features):
        features = features.view(-1)
        out = self.layer1(features)
        for layer in self.inner_layers:
            out = layer(out)
        out = self.final_layer(features)
        out = out.expand((self.num_output_processor_nodes, self.processor_node_dim))
        return out



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
        

        return x



class DiffPool(torch.nn.Module):
    def __init__(self, num_input_processor_nodes, num_output_processor_nodes, processor_node_dim, div_by = 1.5):
        super(DiffPool, self).__init__()

        self.num_steps = math.log(num_input_processor_nodes/num_output_processor_nodes)/math.log(1.5)
        self.num_steps = math.floor(self.num_steps)
        num_input_nodes = num_input_processor_nodes

        self.gnn_pools = torch.nn.ModuleList()
        self.gnn_embeds = torch.nn.ModuleList()

        for _ in range(self.num_steps - 1):
            num_output_nodes = math.ceil(num_input_nodes/div_by)
            self.gnn_pools.append(GNN(processor_node_dim, processor_node_dim, num_output_nodes))
            self.gnn_embeds.append(GNN(processor_node_dim, processor_node_dim, processor_node_dim))
            num_input_nodes = num_output_nodes
        
        self.gnn_pools.append(GNN(processor_node_dim, processor_node_dim, num_output_processor_nodes))
        self.gnn_embeds.append(GNN(processor_node_dim, processor_node_dim, processor_node_dim))


    def forward(self, x, adj, mask=None):
        for i in range(self.num_steps):
            s = self.gnn_pools[i](x, adj, mask)
            x = self.gnn_embeds[i](x, adj, mask)

            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0
        return x

