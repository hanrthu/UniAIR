import math
import torch
import torch.nn as nn

from torch_scatter import scatter_max
from torch_cluster import nearest

class SpatialLineGraph(nn.Module):
    """
    Spatial line graph construction module from `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        num_angle_bin (int, optional): number of bins to discretize angles between edges
    """

    def __init__(self, num_angle_bin=8):
        super(SpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = graph.line_graph()
        node_in, node_out = graph.edge_list[:, :2].t()
        edge_in, edge_out = line_graph.edge_list.t()

        # compute the angle ijk
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        vector1 = graph.node_position[node_i] - graph.node_position[node_j]
        vector2 = graph.node_position[node_k] - graph.node_position[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long().clamp(max=self.num_angle_bin - 1)
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)

        return type(line_graph)(edge_list=edge_list, node_feature=line_graph.node_feature, num_node=line_graph.num_node, num_edge=line_graph.num_edge, 
                                num_relation=self.num_angle_bin, edge_weight=line_graph.edge_weight, batch_size=line_graph.batch_size)

class InterfaceGraph(nn.Module):

    def __init__(self, entity_level='node', cutoff=10.0):
        super(InterfaceGraph, self).__init__()
        self.entity_level = entity_level
        self.cutoff = cutoff

    def get_interface(self, a, b):
        nearest_b_indices = nearest(a.node_position, b.node_position, a.node2graph, b.node2graph)
        nearest_distance = (a.node_position - b.node_position[nearest_b_indices]).norm(dim=-1)
        is_interface_atom = nearest_distance < self.cutoff

        is_interface_resiude = scatter_max(is_interface_atom.long(), a.atom2residue)[0]
        is_interface_atom = is_interface_resiude[a.atom2residue].bool()
        return is_interface_atom
    
    def forward(self, graph):
        entity_a = graph.subgraph(graph.entity_a)
        entity_b = graph.subgraph(graph.entity_b)
        interface_mask_a = self.get_interface(entity_a, entity_b)
        interface_mask_b = self.get_interface(entity_b, entity_a)
        mask = torch.zeros(graph.num_node, dtype=torch.bool, device=graph.device)
        mask[graph.entity_a] = interface_mask_a
        mask[graph.entity_b] = interface_mask_b
        mask[graph.is_mutation] = 1

        if self.entity_level in ["node", "atom"]:
            graph = graph.subgraph(mask)
        else:
            graph = graph.subresidue(mask)

        return graph