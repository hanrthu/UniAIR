import torch
import torch.nn.functional as F
from easydict import EasyDict
from ._base import register_transform
from torch_scatter import scatter_max
from .graph import HeteroGraph, Graph
from .variadic import *
from data.protein.residue_constants import lexico_restypes, atom_order

def one_hot(aa: torch.Tensor, length):
    # Num classes of Amino Acids are 21, and the X should be zeros
    num_classes = length
    one_hot = F.one_hot(aa.clamp(0, num_classes-2), num_classes=num_classes-1).float()
    unk_mask = (aa == num_classes-1).unsqueeze(-1)
    one_hot = one_hot * ~unk_mask
    
    return one_hot

@register_transform('pre_gearbind_transform')
class PreGearbindTransform(object):
    def __init__(self, spatial_config=None, knn_config=None, sequential_config=None, **kwargs):
        super().__init__()
        self.spatial_config = spatial_config
        self.knn_config = knn_config
        self.sequential_config = sequential_config
    
    def transform_aa(self, aa, residue_symbol2id):
        map_dict = {}   
        lexico_id2seq = {i: item for i, item in enumerate(lexico_restypes)}
        for key in lexico_id2seq:
            value = lexico_id2seq[key]
            mapping = residue_symbol2id[value]
            map_dict[key] = mapping
        keys = torch.tensor(list(map_dict.keys()), device=aa.device)
        values = torch.tensor(list(map_dict.values()), device=aa.device)
        indices = torch.searchsorted(keys, aa)
        indices = torch.clamp(indices, max=len(keys) - 1)
        mapped_aa = values[indices]
        return mapped_aa
    
    def transform_atom(self, atom37, atom_pos37, atom_name2id):
        atom_dict = {v: k for k,v in atom_order.items()}
        map_dict = {}
        for key in atom_dict:
            value = atom_dict[key]
            mapping = atom_name2id[value]
            map_dict[key] = mapping
        map_dict = {v:k for k, v in map_dict.items()}
        index1 = range(atom37.shape[-1])
        index2 = torch.tensor([map_dict[i] for i in index1], device=atom37.device)
        
        index_atom2 = index2.view(1, 1, atom37.shape[-1]).expand(atom37.shape)
        mapped_atom37 = torch.gather(atom37, dim=2, index=index_atom2)
        index_pos2 = index2.view(1, 1, atom37.shape[-1], 1).expand(atom_pos37.shape)
        mapped_atom_pos37 = torch.gather(atom_pos37, dim=2, index=index_pos2)
        return mapped_atom37, mapped_atom_pos37
    
    def transform_data(self, structure):
        # Indexing system in Gearbind is different from that in others, we need to transfer
        residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
                         "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19, "X": 20}
        
        atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                    "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}
        
        residue2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}
        mapped_aa = self.transform_aa(structure['aa'], residue_symbol2id)
        # Residue level info
        residue_feature = one_hot(mapped_aa.clone(), 22) # (N,L) -> (N,L,D)
        batch_size = residue_feature.shape[0]
        # print("Residue Feature:", residue_feature.shape)
        batch_ids = torch.repeat_interleave(torch.arange(batch_size, device=residue_feature.device), residue_feature.shape[1], dim=0)
        mask_residue = (structure['mask_atoms'][:, :, 1] & structure['mask_atoms'][:, :, 0] & structure['mask_atoms'][:, :, 2]).reshape(-1)
        residue_feature = residue_feature.reshape(-1, residue_feature.shape[-1])
        residue_feature = residue_feature[mask_residue]
        residue_type = mapped_aa.reshape(-1)[mask_residue]
        residue2graph = batch_ids[mask_residue]
        num_residues = (structure['mask_atoms'][:, :, 1] & structure['mask_atoms'][:, :, 0] & structure['mask_atoms'][:, :, 2]).sum(dim=-1)
        num_residue = mask_residue.sum()
        # num_cum_residues = torch.cat([torch.tensor([0,], device=num_residues.device), torch.cumsum(num_residues, dim=0)[:-1]])
        num_cum_residues = torch.cumsum(num_residues, dim=0)
        chain_nb = structure['chain_nb'] # (N, L)
        offsets = torch.max(chain_nb, dim=1, keepdim=True)[0] + 1
        cum_offsets = torch.cat([torch.tensor([0,], device=offsets.device).unsqueeze(1), torch.cumsum(offsets, dim=0)[:-1]])
        chain_nb_cum = (chain_nb + cum_offsets).reshape(-1)[mask_residue]

        # Atom level info
        # atom37 = structure['atom37'] #(N,L,37)
        mapped_atom37, mapped_atom_pos37 = self.transform_atom(structure['atom_mask37'], structure['atom_pos37'], atom_name2id)
        atom37 = mapped_atom37.reshape(-1, structure['atom_mask37'].shape[-1]) #(N*L,37)
        atom37 = atom37[mask_residue]
        # atom_pos37 = structure['atom_pos37']
        atom_pos37 = mapped_atom_pos37.reshape(-1, structure['atom_pos37'].shape[-2], 3) #(N*L,37,3)
        atom_pos37 = atom_pos37[mask_residue]
        indices = torch.nonzero(atom37, as_tuple=True)
        atom_idx = indices[1]
        atom2residue = indices[0]
        node_position = atom_pos37[atom2residue, atom_idx]
        atom2graph = residue2graph[atom2residue]
        num_nodes = atom2graph.bincount(minlength=structure['aa'].shape[0])
        num_node = len(atom_idx)
        atom_feature = torch.cat([
            one_hot(atom_idx, 38),
            residue_feature[atom2residue]
        ], dim=1)
        graph = Graph(
            node_feature = atom_feature,
            node_position = node_position,
            num_node = num_node, # This is the total count
            num_nodes = num_nodes, # This is the bincount
            atom_name = atom_idx,
            atom_name2id = atom_name2id,
            residue2id = residue2id,
            atom2residue = atom2residue,
            atom2graph = atom2graph,
            node2graph = atom2graph,
            residue_feature = residue_feature,
            residue_type = residue_type,
            batch_size = batch_size,
            
            residue2graph = residue2graph,
            num_residue= num_residue, # This is the total count
            num_residues = num_residues, # This is the bincount
            mask_residue = mask_residue,
            num_cum_residues = num_cum_residues,

            chain_id = chain_nb_cum
        )
        
        return graph
        
    def __call__(self, data):
        wild_type = data['complex_wt']
        wt_graph = self.transform_data(wild_type)

        graph_transform = HeteroGraph(spatial_config=self.spatial_config,
                                    knn_config=self.knn_config,
                                    sequential_config=self.sequential_config)
        
        wt_graph = graph_transform(wt_graph)
        wt_graph.edge_weight = torch.ones(wt_graph.num_edge, device=wt_graph.edge_list.device)
        graph = EasyDict({
            'wild_type': wt_graph,
            'labels': data['labels'],
            'label_mask': data['label_mask'],
            'complex': data['complex'],
        })
        
        if 'complex_mut' in data:
            mutant = data['complex_mut']
            mut_graph = self.transform_data(mutant)
            mut_graph = graph_transform(mut_graph)
            mut_graph.edge_weight = torch.ones(mut_graph.num_edge, device=mut_graph.edge_list.device)
            graph['mutant'] = mut_graph
       
        return graph