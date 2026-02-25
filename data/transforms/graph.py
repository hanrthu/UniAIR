import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import register_transform
from torch_cluster import knn_graph, radius_graph
from .variadic import variadic_meshgrid
import warnings

def one_hot(aa: torch.Tensor, length):
    # Num classes of Amino Acids are 21, and the X should be zeros
    num_classes = length
    one_hot = F.one_hot(aa.clamp(0, num_classes-2), num_classes=num_classes-1).float()
    unk_mask = (aa == num_classes-1).unsqueeze(-1)
    one_hot = one_hot * ~unk_mask
    
    return one_hot

class SpatialEdge(object):
    """
    Construct edges between nodes within a specified radius.

    Parameters:
        radius (float, optional): spatial radius
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, radius=5, min_distance=5, max_distance=None, max_num_neighbors=32):
        super(SpatialEdge, self).__init__()
        self.radius = radius
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, graph):
        """
        Return spatial radius edges constructed based on the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = radius_graph(graph.node_position, r=self.radius, batch=graph.node2graph, max_num_neighbors=self.max_num_neighbors).t()
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=graph.node_position.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)

        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        if self.max_distance:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() > self.max_distance
            edge_list = edge_list[~mask]
        
        node_in, node_out = edge_list.t()[:2]
        mask = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, 1
    
class KNNEdge(object):
    """
    Construct edges between each node and its nearest neighbors.

    Parameters:
        k (int, optional): number of neighbors
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, k=10, min_distance=5, max_distance=None):
        super(KNNEdge, self).__init__()
        self.k = k
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(self, graph):
        """
        Return KNN edges constructed from the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = knn_graph(graph.node_position, k=self.k, batch=graph.node2graph).t()
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=graph.node_position.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)

        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        if self.max_distance:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() > self.max_distance
            edge_list = edge_list[~mask]

        node_in, node_out = edge_list.t()[:2]
        mask = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, 1
    
# 这里有个问题，interface上的序列序号并不连续
class SequentialEdge(object):
    """
    Construct edges between atoms within close residues.

    Parameters:
        max_distance (int, optional): maximum distance between two residues in the sequence
    """

    def __init__(self, max_distance=2, only_backbone=False):
        super(SequentialEdge, self).__init__()
        self.max_distance = max_distance
        self.only_backbone = only_backbone

    def __call__(self, graph):
        """
        Return sequential edges constructed based on the input graph.
        Edge types are defined by the relative distance between two residues in the sequence

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        atom_name2id ={'N': 0, 'CA': 1, 'C': 2, 'CB': 3, 'O': 4}
        if self.only_backbone:
            is_backbone = (graph.atom_name == atom_name2id["CA"]) \
                        | (graph.atom_name == atom_name2id["C"]) \
                        | (graph.atom_name == atom_name2id["N"])
            atom2residue: torch.Tensor = graph.atom2residue[is_backbone]
        else:
            atom2residue: torch.Tensor = graph.atom2residue
        residue2num_atom = atom2residue.bincount(minlength=graph.num_residue)
        edge_list = []
        for i in range(-self.max_distance, self.max_distance + 1):
            node_index = torch.arange(graph.num_node, device=graph.node_position.device)
            residue_index = torch.arange(graph.num_residue, device=graph.node_position.device)
            if i > 0:
                is_node_in = graph.atom2residue < graph.num_cum_residues[graph.atom2graph] - i
                is_node_out = graph.atom2residue >= (graph.num_cum_residues - graph.num_residues)[graph.atom2graph] + i
                is_residue_in = residue_index < graph.num_cum_residues[graph.residue2graph] - i
                is_residue_out = residue_index >= (graph.num_cum_residues - graph.num_residues)[graph.residue2graph] + i
            else:
                is_node_in = graph.atom2residue >= (graph.num_cum_residues - graph.num_residues)[graph.atom2graph] - i
                is_node_out = graph.atom2residue < graph.num_cum_residues[graph.atom2graph] + i
                is_residue_in = residue_index >= (graph.num_cum_residues - graph.num_residues)[graph.residue2graph] - i
                is_residue_out = residue_index < graph.num_cum_residues[graph.residue2graph] + i
            if self.only_backbone:    
                is_node_in = is_node_in & is_backbone
                is_node_out = is_node_out & is_backbone
            node_in = node_index[is_node_in]
            node_out = node_index[is_node_out]
            # group atoms by residue ids
            node_in = node_in[graph.atom2residue[node_in].argsort()]
            node_out = node_out[graph.atom2residue[node_out].argsort()]
            num_node_in = residue2num_atom[is_residue_in]
            num_node_out = residue2num_atom[is_residue_out]
            node_in, node_out = variadic_meshgrid(node_in, num_node_in, node_out, num_node_out)
            # exclude cross-chain edges
            is_same_chain = (graph.chain_id[graph.atom2residue[node_in]] == graph.chain_id[graph.atom2residue[node_out]])
            node_in = node_in[is_same_chain]
            node_out = node_out[is_same_chain]
            relation = torch.ones(len(node_in), dtype=torch.long, device=node_in.device) * (i + self.max_distance)
            edges = torch.stack([node_in, node_out, relation], dim=-1)
            edge_list.append(edges)

        edge_list = torch.cat(edge_list)

        return edge_list, 2 * self.max_distance + 1

class Graph(object):
    def __init__(self,
                node_feature=None,
                atom_name2id=None,
                residue2id=None,
                node_position=None,
                atom_name=None,
                residue_feature=None,
                residue_type=None,
                atom2residue=None,
                residue2graph=None,
                atom2graph=None,
                node2graph=None,
                mask_residue=None,
                num_residue=None,
                num_residues=None,
                num_cum_residues=None,
                num_node=None,
                num_nodes=None,
                chain_id=None,
                edge_list=None,
                num_edge=None,
                edge_weight=None,
                num_relation=None,
                batch_size=None,
                **kwargs):
        super().__init__()
        self.node_feature = node_feature
        self.atom_name2id = atom_name2id
        self.residue2id = residue2id
        self.node_position = node_position
        self.atom_name = atom_name
        self.residue_feature = residue_feature
        self.residue_type = residue_type
        self.atom2residue = atom2residue
        self.residue2graph = residue2graph
        self.atom2graph = atom2graph
        self.node2graph = node2graph
        self.mask_residue = mask_residue
        self.num_residue = num_residue
        self.num_residues = num_residues
        self.num_cum_residues = num_cum_residues
        self.num_node = num_node
        self.num_nodes = num_nodes
        self.chain_id = chain_id
        self.edge_list = edge_list
        self.num_edge = num_edge
        self.edge_weight = edge_weight
        self.num_relation = num_relation
        self.batch_size = batch_size
    
    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = torch.arange(start, stop, step, device=index.device)
        else:
            index = torch.as_tensor(index, device=index.device)
            if index.ndim == 0:
                index = index.unsqueeze(0)
            if index.dtype == torch.bool:
                if index.shape != (count,):
                    raise IndexError("Invalid mask. Expect mask to have shape %s, but found %s" %
                                     ((int(count),), tuple(index.shape)))
                index = index.nonzero().squeeze(-1)
            else:
                index = index.long()
                max_index = -1 if len(index) == 0 else index.max().item()
                if max_index >= count:
                    raise IndexError("Invalid index. Expect index smaller than %d, but found %d" % (count, max_index))
        return index
    
    def line_graph(self):
        """
        Construct a line graph of this graph.
        The node feature of the line graph is inherited from the edge feature of the original graph.

        In the line graph, each node corresponds to an edge in the original graph.
        For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
        there is a directed edge (a, b) -> (b, c) in the line graph.

        Returns:
            Graph
        """
        node_in, node_out = self.edge_list.t()[:2]
        edge_index = torch.arange(self.num_edge, device=node_in.device)
        edge_in = edge_index[node_out.argsort()]
        edge_out = edge_index[node_in.argsort()]

        degree_in = node_in.bincount(minlength=self.num_node)
        degree_out = node_out.bincount(minlength=self.num_node)
        size = degree_out * degree_in
        starts = (size.cumsum(0) - size).repeat_interleave(size)
        range = torch.arange(size.sum(), device=node_in.device)
        # each node u has degree_out[u] * degree_in[u] local edges
        local_index = range - starts
        local_inner_size = degree_in.repeat_interleave(size)
        edge_in_offset = (degree_out.cumsum(0) - degree_out).repeat_interleave(size)
        edge_out_offset = (degree_in.cumsum(0) - degree_in).repeat_interleave(size)
        edge_in_index = torch.div(local_index, local_inner_size, rounding_mode="floor") + edge_in_offset
        edge_out_index = local_index % local_inner_size + edge_out_offset

        edge_in = edge_in[edge_in_index]
        edge_out = edge_out[edge_out_index]
        edge_list = torch.stack([edge_in, edge_out], dim=-1)
        node_feature = getattr(self, "edge_feature", None)
        num_node = self.num_edge
        num_edge = size.sum()
        edge_weight= torch.ones(num_edge, device=edge_list.device)
        return Graph(edge_list=edge_list, num_node=num_node, num_edge=num_edge, node_feature=node_feature, edge_weight=edge_weight, batch_size=self.batch_size)
        

class HeteroGraph(object):
    max_seq_dist = 10
    def __init__(self, spatial_config=None, knn_config=None, sequential_config=None, **kwargs):
        super().__init__()
        self.edge_layers = []
        if spatial_config is not None:
            spatial_layer = SpatialEdge(**spatial_config)
            self.edge_layers.append(spatial_layer)
        if knn_config is not None:
            knn_layer = KNNEdge(**knn_config)
            self.edge_layers.append(knn_layer)
        if sequential_config is not None:
            sequential_layer = SequentialEdge(**sequential_config)
            self.edge_layers.append(sequential_layer)
    
    def edge_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]
        sequential_dist = torch.abs(residue_in - residue_out)
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        return torch.cat([
            one_hot(in_residue_type, len(graph.residue2id) + 1),
            one_hot(out_residue_type, len(graph.residue2id) + 1),
            one_hot(r, num_relation + 1),
            one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 2),
            spatial_dist.unsqueeze(-1)
        ], dim=-1)
    
    def __call__(self, graph):
        edge_list = []
        num_edges = []
        num_relations = []
        device = 'cpu'
        for layer in self.edge_layers:
            edges, num_relation = layer(graph)
            edge_list.append(edges)
            num_edges.append(len(edges))
            num_relations.append(num_relation)
            device = edges.device
            
        edge_list = torch.cat(edge_list)
        num_edges = torch.as_tensor(num_edges, device=device)
        num_relations = torch.as_tensor(num_relations, device=device)
        num_relation = num_relations.sum()
        num_edge = num_edges.sum()
        offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        edge_list[:, 2] += offsets
        graph.edge_index = edge_list[:, :2].t() # [2, |E|]
        graph.edge_relations = edge_list[:, 2]
        graph.num_relation = num_relation
        graph.edge_list = edge_list
        graph.num_edge = num_edge
        graph.edge_feature = self.edge_gearnet(graph, edge_list, num_relation)

        return graph