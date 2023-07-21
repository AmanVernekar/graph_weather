from .forecast import GraphWeatherForecaster
import torch
from typing import Optional
from torch import nn

class ParallelForecaster(torch.nn.Module):
    def __init__(
        self,
        lat_lons: list,
        num_steps:int = 2,
        resolution: int = 2,
        feature_dim: int = 42, #TODO change back to 78
        aux_dim: int = 0, #TODO change back to 24
        output_dim: Optional[int] = None,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_blocks: int = 3, #change back to 3
        hidden_dim_processor_node: int = 128,
        hidden_dim_processor_edge: int = 128,
        hidden_layers_processor_node: int = 2, #change all 1 hidden layers back to 2
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 64,
        hidden_layers_decoder: int = 2,
        #norm_type = None,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim

        self.num_steps = num_steps
        self.models = []
        for i in range(num_steps):
            self.models.append(
                GraphWeatherForecaster(
                    lat_lons=lat_lons,
                    resolution=resolution,
                    feature_dim=feature_dim,
                    aux_dim=aux_dim,
                    output_dim=output_dim,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    num_blocks=num_blocks,
                    hidden_dim_processor_node=hidden_dim_processor_node,
                    hidden_dim_processor_edge=hidden_dim_processor_edge,
                    hidden_layers_processor_node=hidden_layers_processor_node,
                    hidden_layers_processor_edge=hidden_layers_processor_edge,
                    hidden_dim_decoder=hidden_dim_decoder,
                    hidden_layers_decoder=hidden_layers_decoder,
                    norm_type=norm_type,
                    use_checkpointing=use_checkpointing
                ).to(torch.device('cuda'))
            )
        self.final_layer = nn.Linear(num_steps*output_dim, output_dim)
    

    def forward(self, features):
        out = []
        for i in range(self.num_steps):
            inp = torch.stack([features[0][i]])
            out.append(self.models[i](inp.to(features.device)))
        return self.final_layer(torch.cat(out, dim=-1))