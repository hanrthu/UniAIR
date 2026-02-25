import copy
import random
from typing import Literal, Any, Iterable, Optional, Callable, Union
from pathlib import Path
from functools import partial
from collections import Counter

import torch
import torch_geometric.transforms as T
import einops
from typing import Dict
import torch.nn.functional as F

from ._base import register_transform

def contains_nan_or_inf(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

DDG_INFERENCE_TYPE = Literal[
    'masked_marginals',
    'wt_marginals',  # basline
    'embedding_difference',  # baseline
    'embedding_concatenation'  # baseline
]

def one_hot(aa: torch.Tensor):
    # Num classes of Amino Acids are 21, and the X should be zeros
    from data.protein.residue_constants import restypes_with_x
    num_classes = len(restypes_with_x)
    one_hot = F.one_hot(aa.clamp(0, num_classes-2), num_classes=num_classes-1).float()
    unk_mask = (aa == num_classes-1).unsqueeze(-1)
    one_hot = one_hot * ~unk_mask
    
    return one_hot

def cal_virtual_c_beta(pos_atoms: torch.Tensor):
    N = pos_atoms[:, :, 0, :]
    Ca = pos_atoms[:, :, 1, :]
    C = pos_atoms[:, :, 2, :]
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=2)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
    vec = Cb - Ca
    
    return vec

def cal_intra_inter_edges(group_id):
    eq_matrix = (group_id.unsqueeze(1) == group_id.unsqueeze(2)).int()
    zero_mask = (group_id != 0).unsqueeze(1) & (group_id != 0).unsqueeze(2)
    edges = eq_matrix * zero_mask.int()
    return edges


@register_transform('mask_mutation_transform')
class MaskMutationTransform(object):
    def __init__(
        self,
        vocab_size = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
    def __call__(self, data: Dict, masked_features: Optional[torch.Tensor] = None):
        if 'mut_flag' in data:
            mut_flag = data['mut_flag']
        else:
            mut_flag = torch.zeros(data['aa'].shape, device=data['aa'].device)
        if torch.sum(mut_flag) == 0:
            data['aa_masked'] = one_hot(data['aa'].clone())
            return data
        # mut_flag = data['mut_flag']
        if masked_features is None:
            masked_features = torch.zeros((torch.sum(mut_flag), self.vocab_size))
        masked_features = masked_features.to(data['aa'].device)
        f_masked = one_hot(data['aa'].clone())
        f_masked[mut_flag, :] = masked_features
        data['aa_masked'] = f_masked
        return data

@register_transform('pre_equiformer_transform')
class PreEquiformerTransform(object):
    def __init__(
        self,
        resolution: str = 'backbone+CB',
        coord_fill_value: float = 0.00001,
        divide_coords_by: float = 4.0,
        intra_inter_edge_features: bool = True,
        mask_mutation_transform: bool = True
    ):
        super().__init__()
        self.coord_fill_value = coord_fill_value
        self.resolution = resolution
        self.divide_coords_by = divide_coords_by
        self.intra_inter_edge_features = intra_inter_edge_features
        self.mask_mutation_transform = mask_mutation_transform

    def __call__(self, data: Dict) -> tuple:
        if self.resolution == 'backbone':
            data['pos_atoms'] = data['pos_atoms'][:, :, :4].contiguous()
            data['mask_atoms'] = data['mask_atoms'][:, :, :4].contiguous()
            data['bfactor_atoms'] = data['bfactor_atoms'][:, :, :4].contiguous()
        elif self.resolution == 'backbone+CB':
            data['pos_atoms'] = data['pos_atoms'][:, :, :5].contiguous()
            data['mask_atoms'] = data['mask_atoms'][:, :, :5].contiguous()
            data['bfactor_atoms'] = data['bfactor_atoms'][:, :, :5].contiguous()
        # Init type-0 features
        # feats_0 = to_dense_batch(data.f, data.batch)[0]
        if self.mask_mutation_transform:
            trans = MaskMutationTransform(vocab_size=20)
            data = trans(data)
            feats_0: torch.Tensor = data['aa_masked']
        else:
            feats_0: torch.Tensor = one_hot(data['aa'].clone())
        # feats_0 = einops.rearrange(feats_0, 'b n d -> b n d 1')
        feats_0 = feats_0.unsqueeze(-1)

        # Init type-1 features by virtual_c_beta_vector
        feats_1 = cal_virtual_c_beta(data['pos_atoms'])
        feats_1 = torch.unsqueeze(feats_1, dim=-2)

        # Init input fiber
        feats = {0: feats_0, 1: feats_1}

        # Init coords and sequence padding mask
        coors, mask = data['pos_atoms'][:, :, 1].contiguous(), data['mask_atoms'][:, :, 1].contiguous()

        # Rescale
        coors /= self.divide_coords_by

        # Convert types
        feats = {t: f.float() for t, f in feats.items()}
        coors = coors.float()

        # Init inter/intra binary edge features
        if self.intra_inter_edge_features:
            edges = cal_intra_inter_edges(data['group_id'])
            edges = edges.to(coors.device)
        else:
            edges = None

        # TODO: Check why is missing of 'pdbcode' error here.
        # pdbcode = data['pdbcode']
        # # Validate
        # for deg, feat in feats.items():
        #     assert not contains_nan_or_inf(feat), f'feats[{deg}] contains NaN or Inf in {pdbcode}'
        # assert not contains_nan_or_inf(coors), f'coords contains NaN or Inf in {pdbcode}'
        # assert not contains_nan_or_inf(mask), f'mask contains NaN or Inf in {pdbcode}'

        return dict(inputs=feats, coors=coors, mask=mask, edges=edges)
    