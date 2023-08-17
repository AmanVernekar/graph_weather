from .forecast import GraphWeatherForecaster
from .time_attention import ParallelForecaster
from graph_weather.models import Graph2Vec, DiffPool
import torch
from typing import Optional
import h3
from torch_geometric import utils

class MultiResoForecaster(torch.nn.Module):
    def __init__(
        self,
        lat_lons_list: list, # list of three sets of lat_lons (for earth, europe and UK)
        # model_type: str,
        # num_steps:int = 2,
        resolutions = [2, 3, 4],
        feature_dim: int = 42, #TODO change back to 78
        aux_dim: int = 0, #TODO change back to 24
        output_dim: Optional[int] = None,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_blocks: int = 6, #change back to 3
        hidden_dim_processor_node: int = 128,
        hidden_dim_processor_edge: int = 128,
        hidden_layers_processor_node: int = 2, #change all 1 hidden layers back to 2
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 64,
        hidden_layers_decoder: int = 2,
        #norm_type = None,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        attention_type = None,
        num_node_heads = 8,
        num_edge_heads = 8
    ):
        super().__init__()
        global_lat_lons = lat_lons_list[0]
        europe_lat_lons = lat_lons_list[1]
        uk_lat_lons = lat_lons_list[2]

        global_res = resolutions[0]
        europe_res = resolutions[1]
        uk_res = resolutions[2]

        # create three separate GWF models. Don't think about their internal functions or output here. 
        # Just set regional = True or False as required and deal with the internal workings of each model later

        self.attention_type = attention_type
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim
        
        self.global_model = GraphWeatherForecaster(
                    lat_lons=global_lat_lons,
                    resolution=global_res,
                    regional=False,
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
                    use_checkpointing=use_checkpointing,
                    is_last_model=False
                )
        
        self.europe_model = GraphWeatherForecaster(
                    lat_lons=europe_lat_lons,
                    resolution=europe_res,
                    regional=True,
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
                    use_checkpointing=use_checkpointing,
                    attention_type=attention_type,
                    num_node_heads=num_node_heads,
                    num_edge_heads=num_edge_heads,
                    is_last_model=False
                )
        
        self.uk_model = GraphWeatherForecaster(
                    lat_lons=uk_lat_lons,
                    resolution=uk_res,
                    regional=True,
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
                    use_checkpointing=use_checkpointing,
                    attention_type=attention_type,
                    num_node_heads=num_node_heads,
                    num_edge_heads=num_edge_heads,
                    is_last_model=True
                )
        
        num_global_processor_nodes = len(self.global_model.encoder.base_h3_grid)
        num_europe_processor_nodes = len(self.europe_model.encoder.base_h3_grid)
        num_uk_processor_nodes = len(self.uk_model.encoder.base_h3_grid)

        if attention_type == 'vector':
            self.pool1 = Graph2Vec(
                num_input_processor_nodes=num_global_processor_nodes,
                num_output_processor_nodes=num_europe_processor_nodes,
                processor_node_dim=hidden_dim_processor_node
            )
            self.pool2 = Graph2Vec(
                num_input_processor_nodes=num_europe_processor_nodes,
                num_output_processor_nodes=num_uk_processor_nodes,
                processor_node_dim=hidden_dim_processor_node
            )
        
        elif attention_type == 'diff_pool':
            self.pool1 = DiffPool(
                num_input_processor_nodes=num_global_processor_nodes,
                num_output_processor_nodes=num_europe_processor_nodes,
                processor_node_dim=hidden_dim_processor_node,
                div_by=1.5
            )
            self.pool2 = DiffPool(
                num_input_processor_nodes=num_europe_processor_nodes,
                num_output_processor_nodes=num_uk_processor_nodes,
                processor_node_dim=hidden_dim_processor_node,
                div_by=1.5
            )
            self.adj1 = utils.to_dense_adj(self.global_model.encoder.edge_idx)
            self.adj2 = utils.to_dense_adj(self.europe_model.encoder.edge_idx)
        
    
    def forward(self, features):
        if self.attention_type == 'vector':
            out = self.global_model(features[0])
            out = self.pool1(out)
            out = self.europe_model(features[1], out)
            out = self.pool2(out)
            out = self.uk_model(features[2], out)
        
        elif self.attention_type == 'diff_pool':
            out = self.global_model(features[0])
            out = self.pool1(out, self.adj1)
            out = self.europe_model(features[1], out)
            out = self.pool2(out, self.adj2)
            out = self.uk_model(features[2], out)
        
        return out

