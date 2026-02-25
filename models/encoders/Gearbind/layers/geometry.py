import torch
import torch.nn as nn
from torch_scatter import scatter_add
from data.transforms.graph import Graph
class AtomPositionGather(nn.Module):

    def from_3_points(self, p_x_axis, origin, p_xy_plane, eps=1e-10):
        """
            Adpated from torchfold
            Implements algorithm 21. Constructs transformations from sets of 3 
            points using the Gram-Schmidt algorithm.
            Args:
                x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_x_axis = torch.unbind(p_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(p_x_axis, origin)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        return rots

    def forward(self, graph):
        residue_mask = \
            scatter_add((graph.atom_name == graph.atom_name2id["N"]).float(), graph.atom2residue, dim_size=graph.num_residue) + \
            scatter_add((graph.atom_name == graph.atom_name2id["CA"]).float(), graph.atom2residue, dim_size=graph.num_residue) + \
            scatter_add((graph.atom_name == graph.atom_name2id["C"]).float(), graph.atom2residue, dim_size=graph.num_residue)
        residue_mask = (residue_mask == 3)
        atom_mask = residue_mask[graph.atom2residue] & (graph.atom_name == graph.atom_name2id["CA"])
        # graph = graph.subresidue(residue_mask)

        atom_pos = torch.full((graph.num_residue, len(graph.atom_name2id), 3), float("inf"), dtype=torch.float, device=graph.node_position.device)
        atom_pos[graph.atom2residue, graph.atom_name] = graph.node_position
        atom_pos_mask = torch.zeros((graph.num_residue, len(graph.atom_name2id)), dtype=torch.bool, device=graph.node_position.device)
        atom_pos_mask[graph.atom2residue, graph.atom_name] = 1

        node_position = atom_pos[:, 1, :]
        subgraph = Graph(num_nodes=graph.num_residues, node_position=node_position, node2graph=graph.residue2graph, atom_name2id=graph.atom_name2id)
        # graph = graph.subgraph()
        frame = self.from_3_points(
            atom_pos[:, graph.atom_name2id["N"]],
            atom_pos[:, graph.atom_name2id["CA"]],
            atom_pos[:, graph.atom_name2id["C"]]
        ).transpose(-1, -2)

        subgraph.view = 'residue'
        subgraph.atom_pos = atom_pos
        subgraph.atom_pos_mask = atom_pos_mask
        subgraph.frame = frame

        return subgraph, atom_mask
