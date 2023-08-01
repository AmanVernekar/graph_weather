from .forecast import GraphWeatherForecaster
import torch
from typing import Optional
from torch import nn

class ParallelForecaster(torch.nn.Module):
    def __init__(
        self,
        lat_lons: list,
        model_type: str,
        num_steps:int = 2,
        regional: bool = False,
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
        self.models = nn.ModuleList()
        for i in range(num_steps):
            module = GraphWeatherForecaster(
                    lat_lons=lat_lons,
                    resolution=resolution,
                    regional=regional,
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
                )
            self.models.append(module=module)

        self.model_type = model_type

        if model_type == 'linear':
            self.final_layer = nn.Linear(num_steps*output_dim, output_dim)
        elif model_type == 'params':
            self.params = nn.ParameterList([nn.Parameter(data=torch.tensor([1/self.num_steps])) for i in range(self.num_steps)])
        elif model_type == 'simple_attention':
            self.attention_layer = nn.Linear(num_steps*output_dim, num_steps, bias=False)
            self.leaky = nn.LeakyReLU(0.2) 
            self.soft = nn.Softmax(dim=1)   

    def forward(self, features):
        features = features[0]
        
        if self.model_type == 'linear':
            out = [self.models[i](torch.stack([features[i]]).to(features.device)) for i in range(self.num_steps)]
            return self.final_layer(torch.cat(out, dim=-1))

        elif self.model_type == 'params':
            out = torch.zeros(features[0].shape[0], features[0].shape[1]).to(features.device)
            for i in range(self.num_steps):
                out += self.params[i]*self.models[i](torch.stack([features[i]]).to(features.device))[0]
            return torch.stack([out])
        
        elif self.model_type == 'simple_attention':
            embedding = [self.models[i](torch.stack([features[i]]).to(features.device))[0] for i in range(self.num_steps)]
            alphas = self.soft(self.leaky(self.attention_layer(torch.cat(embedding, dim=-1))))
            alphas = torch.unsqueeze(alphas.T, dim=-1)

            out = torch.mul(alphas, torch.stack(embedding))
            out = torch.sum(out, dim=0)
            return out
            
            # out = torch.zeros(features[0].shape[0], features[0].shape[1]).to(features.device) 

            # # lets say we have M alpahs, and N steps
            # # you are broadcasting the M alphas to a tensor of size M*N


            # for i, coeffs in enumerate(alphas):
            #     for j in range(self.num_steps):
            #         out[i] = out[i] + coeffs[j]*embedding[j][i]
            
            # return torch.stack([out])
