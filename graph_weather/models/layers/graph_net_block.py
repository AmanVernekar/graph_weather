"""
Functions for building GNN

This code is taken from https://github.com/CCSI-Toolset/MGN which is available under the
US Government License
"""
from typing import Optional, Tuple

import torch
from torch import cat, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum
import math


class MLP(nn.Module):
    """MLP for graph processing"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128, #TODO: change all 128 back to 128
        hidden_dim: int = 128,
        hidden_layers: int = 2, #TODO: change all 1 hidden_layers back to 2
        norm_type: Optional[str] = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        """
        MLP

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type one of 'LayerNorm', 'GraphNorm',
                'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing or not
        """

        super(MLP, self).__init__()
        self.use_checkpointing = use_checkpointing

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP

        Args:
            x: Node or edge features

        Returns:
            The transformed tensor
        """
        if self.use_checkpointing:
            out = checkpoint(self.model, x, use_reentrant=False)
        else:
            out = self.model(x)
        return out


#############################

# issue with MessagePassing class:
# Only node features are updated after MP iterations
# Need to use MetaLayer to also allow edge features to update


class EdgeProcessor(nn.Module):
    """EdgeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        Edge processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layers
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge, in_dim_edge, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor, u=None, batch=None
    ) -> torch.Tensor:
        """
        Compute the edge part of the message passing

        Args:
            src: Source node tensor
            dest: Destination node tensor
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch Ids, ignored

        Returns:
            The updated edge attributes
        """
        out = cat(
            [src, dest, edge_attr], -1
        )  # concatenate source node, destination node, and edge embeddings
        out = self.edge_mlp(out)
        # out += edge_attr  # residual connection

        return out


class NodeProcessor(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u=None, batch=None
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        row, col = edge_index
        out = scatter_sum(edge_attr, col, dim=0)  # aggregate edge message by target
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        # out += x  # residual connection

        return out


def build_graph_processor_block(
    in_dim_node: int = 128,
    in_dim_edge: int = 128,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    norm_type: str = "LayerNorm",
) -> torch.nn.Module:
    """
    Build the Graph Net Block

    Args:
        in_dim_node: Input node feature dimension
        in_dim_edge: Input edge feature dimension
        hidden_dim_node: Number of nodes in hidden layer for graph node processing
        hidden_dim_edge: Number of nodes in hidden layer for graph edge processing
        hidden_layers_node: Number of hidden layers for node processing
        hidden_layers_edge: Number of hidden layers for edge processing
        norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    Returns:
        torch.nn.Module for the graph processing block
    """

    return MetaLayer(
        edge_model=EdgeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type
        ),
        node_model=NodeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type
        ),
    )


class GraphProcessor(nn.Module):
    """Overall graph processor"""

    def __init__(
        self,
        mp_iterations: int = 15, #change back to 15
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        norm_type: str = "LayerNorm",
        attention_type = None,
        num_node_heads = 8,
        num_edge_heads = 8
    ):
        """
        Graph Processor

        Args:
            mp_iterations: number of message-passing iterations (graph processor blocks)
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim_node: Number of nodes in hidden layers for node processing
            hidden_dim_edge: Number of nodes in hidden layers for edge processing
            hidden_layers_node: Number of hidden layers for node processing
            hidden_layers_edge: Number of hidden layers for edge processing
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(GraphProcessor, self).__init__()
        self.attention_type = attention_type
        self.num_node_heads = num_node_heads
        self.num_edge_heads = num_edge_heads
        self.dim_node_k = in_dim_node/num_node_heads
        self.dim_edge_k = in_dim_edge/num_edge_heads

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type,
                )
            )
        
        if attention_type == 'vector':
            self.node_q_heads = nn.ModuleList()
            self.node_k_heads = nn.ModuleList()
            self.node_v_heads = nn.ModuleList()
            # self.edge_q_heads = nn.ModuleList()
            # self.edge_k_heads = nn.ModuleList()
            # self.edge_v_heads = nn.ModuleList()

            for _ in range(num_node_heads):
                self.node_q_heads.append(nn.Linear(in_dim_node, in_dim_node/num_node_heads, False))
                self.node_k_heads.append(nn.Linear(in_dim_node, in_dim_node/num_node_heads, False))
                self.node_v_heads.append(nn.Linear(in_dim_node, in_dim_node/num_node_heads, False))
            # for _ in range(num_edge_heads):
            #     self.edge_q_heads.append(nn.Linear(in_dim_edge, in_dim_edge/num_edge_heads, False))
            #     self.edge_k_heads.append(nn.Linear(in_dim_edge, in_dim_edge/num_edge_heads, False))
            #     self.edge_v_heads.append(nn.Linear(in_dim_edge, in_dim_edge/num_edge_heads, False))
            
            self.soft = nn.Softmax(1)
            self.final_node_proj = nn.Linear(in_dim_node, in_dim_node, False)            
            # self.final_edge_proj = nn.Linear(in_dim_edge, in_dim_edge, False)
        

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, attention_matrix=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updates to the graph in message passing method

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes

        Returns:
            Updated nodes and edge attributes
        """
        self.attention_matrix = attention_matrix
        for block in self.blocks:
            out_x, out_edge_attr, _ = block(x, edge_index, edge_attr)
            if self.attention_type == 'vector':
                out_node = []
                # out_edge = []
                
                for i in range(self.num_node_heads):
                    out_node.append(self.attention(head_num=i, node=True, x=out_x))
                # for i in range(self.num_edge_heads):
                #     out_edge.append(self.attention(head_num=i, node=False, x=out_edge_attr))

                out_x = torch.cat(out_node, 1)
                # out_edge_attr = torch.cat(out_edge, 1)

                out_x = self.final_node_proj(out_x)
                # out_edge_attr = self.final_edge_proj(out_edge_attr)               

            out_x += x
            out_edge_attr += edge_attr
        return out_x, out_edge_attr
    

    def attention(self, head_num, node, x):
        if node:
            q_matrix = self.node_q_heads[head_num]
            k_matrix = self.node_k_heads[head_num]
            v_matrix = self.node_v_heads[head_num]
            dim_k = self.dim_node_k
        else:
            q_matrix = self.edge_q_heads[head_num]
            k_matrix = self.edge_k_heads[head_num]
            v_matrix = self.edge_v_heads[head_num]
            dim_k = self.dim_edge_k
        
        q = q_matrix(self.attention_matrix)
        k = k_matrix(x)
        v = v_matrix(x)

        out = torch.matmul(q, k.T)/math.sqrt(dim_k)
        out = self.soft(out)
        out = torch.matmul(out, v)
        return out
