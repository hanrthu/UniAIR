from collections.abc import Sequence

import torch
from torch import nn
import sys
from torch_scatter import scatter_add
sys.path.append('Your Path to UniAIR')
from data.transforms.variadic import variadic_to_padded, padded_to_variadic
from models.encoders.Gearbind.layers.readout import SumReadout, MeanReadout
from models.encoders.Gearbind.layers.conv import GeometricRelationalGraphConv
from models.encoders.Gearbind.layers.graph import SpatialLineGraph
from models.encoders.Gearbind.layers.mlp import MultiLayerPerceptron as MLP
from models.encoders.Gearbind.layers.attn import DDGAttention
from models.encoders.Gearbind.layers.geometry import AtomPositionGather
from data.transforms.gearbind import PreGearbindTransform
from models.register import ModelRegister
R = ModelRegister()

@R.register("gearbind")
class BindModel(nn.Module):

    def __init__(self, model_config, pre_transform_cfg, checkpoint=None, num_mlp_layer=2, **kwargs):
        super(BindModel, self).__init__()
        self.model = GearBind(**model_config)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint), strict=True)
            print("Successfully loaded Gearbind Model!")
        self.num_mlp_layer = num_mlp_layer

        hidden_dims = [self.model.output_dim] * (num_mlp_layer - 1)
        self.mlp = MLP(self.model.output_dim * 2, hidden_dims + [1])
        self.transform = PreGearbindTransform(**pre_transform_cfg)
        
    def inference(self, batch, all_loss=None, metric=None):
        return self(batch, all_loss, metric)
    
    def encode(self, batch, all_loss=None, metric=None):
        batch_size = batch['complex_wt']['aa'].shape[0]
        interface_shape = batch['complex_wt']['aa'].shape[1]
        batch = self.transform(batch)
        wild_type = batch["wild_type"]
        wild_type_output = self.model(wild_type, wild_type.node_feature.float(), all_loss=all_loss, metric=metric)
        h = wild_type_output["residue_feature"]
        mask_residue = batch['wild_type'].mask_residue
        expand_h = torch.zeros((batch_size * interface_shape, h.shape[-1]), dtype=h.dtype, device=h.device)
        indices = torch.nonzero(mask_residue, as_tuple=True)[0]
        expand_h[indices] = h 
        h = expand_h.reshape([batch_size, interface_shape, h.shape[1]])
        return h

    def forward_encode(self, batch, all_loss=None, metric=None):
        batch = self.transform(batch)
        mutant = batch["mutant"]
        mutant_output = self.model(mutant, mutant.node_feature.float(), all_loss=all_loss, metric=metric)

        wild_type = batch["wild_type"]
        wild_type_output = self.model(wild_type, wild_type.node_feature.float(), all_loss=all_loss, metric=metric)
        
        residue_wt = wild_type_output["residue_feature"]
        residue_mt =  mutant_output["residue_feature"]
        wild_type_output = wild_type_output['readout_fn'](wild_type_output['residue_graph'], residue_wt)
        mutant_output = mutant_output['readout_fn'](mutant_output['residue_graph'], residue_mt)
        # return (wild_type_output, mutant_output), wild_type_output['residue_graph'], wild_type_output['readout_fn'], mutant_output['residue_graph'], mutant_output['readout_fn']
        embedding = {}
        embedding['ab'] = torch.cat([mutant_output, wild_type_output], dim=-1).squeeze()
        embedding['ba'] = torch.cat([wild_type_output, mutant_output], dim=-1).squeeze()
        return embedding
    def forward_readout(self, h_in):
        pred = self.mlp(h_in['ab'])
        pred = pred - self.mlp(h_in['ba'])
        pred_dict = {
            'y_pred': pred, 
            'wild_type_feature': h_in['ab'][:, :512], 
            'mutant_feature': h_in['ab'][:, 512:]
        }
        return pred_dict, h_in['ab']
    
    def forward(self, batch, all_loss=None, metric=None):
        h_in = self.forward_encode(batch, all_loss, metric)
        pred_dict, _ = self.forward_readout(h_in)
        return pred_dict
      

class GearBind(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum",
                 use_attn=True, **kwargs):
        super(GearBind, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm
        self.use_attn = use_attn

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        self.atom_position_gather = AtomPositionGather()
        if use_attn:
            self.attn_layers = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.attn_layers.append(DDGAttention(self.dims[i+1], self.dims[i+1]))
        if num_angle_bin:
            self.spatial_line_graph = SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norm_layers.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        residue_graph, node_mask = self.atom_position_gather(graph)
        pos_CA, _ = variadic_to_padded(residue_graph.node_position, residue_graph.num_nodes, value=0)
        pos_CB = torch.where(
            residue_graph.atom_pos_mask[:, residue_graph.atom_name2id["CB"], None].expand(-1, 3),
            residue_graph.atom_pos[:, residue_graph.atom_name2id["CB"]],
            residue_graph.atom_pos[:, residue_graph.atom_name2id["CA"]]
        )
        pos_CB, _ = variadic_to_padded(pos_CB, residue_graph.num_nodes, value=0)
        frame, _ = variadic_to_padded(residue_graph.frame, residue_graph.num_nodes, value=0)
        
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        hiddens = []
        layer_input = input
        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * graph.edge_weight.unsqueeze(-1), node_out, dim=0, dim_size=graph.num_node * self.num_relation) 
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norm_layers[i](hidden)

            if self.use_attn:
                x, mask = variadic_to_padded(hidden[node_mask], residue_graph.num_nodes, value=0)
                residue_hidden = self.attn_layers[i](x, pos_CA, pos_CB, frame, mask.bool())
                residue_hidden = padded_to_variadic(residue_hidden, residue_graph.num_nodes)
                hidden[node_mask] += residue_hidden
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        residue_feature = node_feature[node_mask]
        graph_feature = self.readout(residue_graph, node_feature[node_mask])

        return {
            "graph_feature": graph_feature,
            'residue_feature': residue_feature,
            "residue_graph": residue_graph,
            "readout_fn": self.readout
        }
        
        
def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Params:", total_params)
    total_size = total_params * 4
    total_size_MB = total_size / (1024 ** 2)  # MB
    return total_size_MB

if __name__ == '__main__':
    # gearbind = GearBind(
    #     input_dim=58,
    #     hidden_dims=[128, 128, 128, 128],
    #     batch_norm=True,
    #     short_cut=True,
    #     concat_hidden=True,
    #     num_relation=7,
    #     edge_input_dim=59,
    #     num_angle_bin=8,
    #     use_attn=True,
    #     readout='mean'
    #     )
    model_config = {
        'input_dim':58,
        'hidden_dims':[128, 128, 128, 128],
        'batch_norm':True,
        'short_cut':True,
        'concat_hidden':True,
        'num_relation':7,
        'edge_input_dim':59,
        'num_angle_bin':8,
        'use_attn':True,
        'readout':'mean'
    }
    model = BindModel(model_config=model_config, pre_transform_cfg={},checkpoint='./trained_models/gearbind/gearbind.pth')
    print(get_model_size(model))