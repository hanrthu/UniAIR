import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from utils import mask_zero, global_to_local, normalize_vector
from utils import ATOM_CA


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)  # (N, L, *, *)

    logits = torch.where(mask_row, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


def get_neighbors(a, neighbors):
    """
    a: B x L x N x D
    neighbors: B x L x M, M: No. of neighbors
    result: B x L x M x N x D
    """
    B, L, N, D = a.shape
    a = a[:, :, None, :, :].expand(-1, -1, L, -1, -1)
    neighbors = neighbors[:, :, :, None, None].expand(-1, -1, -1, N, D)
    result = a.gather(1, neighbors)
    return result


def get_neighbors_z(a, neighbors):
    """
    a: B x L x L x D
    neighbors: B x L x M, M: No. of neighbors
    result: B x L x M x D
    """
    B, L, LL, D = a.shape
    neighbors = neighbors[:, :, :, None].expand(-1, -1, -1, D)
    result = a.gather(2, neighbors)
    return result


def get_neighbors_p_CB(a, neighbors):
    """
    a: B x L x 3
    neighbors: B x L x M, M: No. of neighbors
    result: B x L x M x 3
    """
    B, L, D = a.shape
    a = a[:, :, None, :].expand(-1, -1, L, -1)
    neighbors = neighbors[:, :, :, None].expand(-1, -1, -1, D)
    result = a.gather(1, neighbors)
    return result


class GeometricEnergyAttention(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim, spatial_attn_mode='CB', value_dim=16, query_key_dim=16,
                 num_query_points=8, num_value_points=8, num_heads=12):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        assert spatial_attn_mode in ('CB')
        self.spatial_attn_mode = spatial_attn_mode

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=False)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=False)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=False)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=False)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (num_heads * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )
        self.layer_norm = nn.LayerNorm(node_feat_dim)

    def _node_logits(self, x, neighbors):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l_knn = get_neighbors(key_l, neighbors)
        logits = torch.einsum('blnd, blmnd->blmn', query_l, key_l_knn)
        return logits

    def _pair_logits(self, z_knn):
        logits_pair = self.proj_pair_bias(z_knn)
        return logits_pair

    def _beta_logits(self, R, t, p_CB, neighbors):
        N, L, _ = t.size()
        q = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        k = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        k_knn = get_neighbors(k, neighbors)
        sum_sq_dist = ((q.unsqueeze(2) - k_knn) ** 2).sum(-1)  # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logtis_beta = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / 9)) / 2)
        return logtis_beta

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x, neighbors):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        value_l_knn = get_neighbors(value_l, neighbors)

        feat_node = alpha.unsqueeze(-1) * value_l_knn  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _beta_aggregation(self, alpha, R, t, p_CB, x, neighbors):
        N, L, _ = t.size()
        M = neighbors.shape[-1]
        v = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3) # (N, L, n_heads, 3)
        v_knn = get_neighbors(v, neighbors)
        aggr = alpha[:, :, :, :, None] * v_knn   # (N, *, L, n_heads, 3)
        aggr = aggr.sum(dim=2)

        feat_points = global_to_local(R, t, aggr)  # (N, L, n_heads, 3)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, p_CB, x, z, mask, neighbors):
        """
          Args:
              R:  Frame basis matrices, (N, L, 3, 3_index).
              t:  Frame external (absolute) coordinates, (N, L, 3).
              x:  Node-wise features, (N, L, F).
              z:  Pair-wise features, (N, L, L, C).
              mask:   Masks, (N, L).
          Returns:
              x': Updated node-wise features, (N, L, F).
          """
        # Attention logits
        z_knn = get_neighbors_z(z, neighbors)
        logits_node = self._node_logits(x, neighbors)
        logits_pair = self._pair_logits(z_knn)
        logits_spatial = self._beta_logits(R, t, p_CB, neighbors)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z_knn)
        feat_node = self._node_aggregation(alpha, x, neighbors)
        feat_spatial = self._beta_aggregation(alpha, R, t, p_CB, x, neighbors)
        feat = torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1)

        # Finally
        feat_all = self.out_transform(feat)  # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm(x + feat_all)
        return x_updated


class GEAEncoderResidue(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, spatial_attn_mode='CB'):
        super().__init__()
        self.blocks = nn.ModuleList([
            GeometricEnergyAttention(node_feat_dim, pair_feat_dim, spatial_attn_mode=spatial_attn_mode)
            for _ in range(num_layers)
        ])

    def forward(self, R, t, p_CB, x, z, mask, neighbors):
        for block in self.blocks:
            x = block(R, t, p_CB, x, z, mask, neighbors)  # Residual connection within the block
        return x


def _alpha_from_logits_atom(logits, atom_mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L, M, 14, 14).
        atom_mask:   Masks, (N, L, 14).
    Returns:
        res_alpha:  Attention weights atom level.  (N, L, M, 14)
        atom_alpha:  Attention weights residue level.  (N, L, M, 14, 14)

    """
    N, L, _, _, _ = logits.size()
    res_mask = atom_mask[:, :, ATOM_CA]
    atom_mask_L = atom_mask.view(N, L, 1, 14, 1).expand_as(logits)  # (N, L, *, *)
    atom_mask_R = atom_mask.view(N, L, 1, 1, 14).expand_as(logits)  # (N, L, *, *)
    atom_mask = atom_mask_L & atom_mask_R
    logits = torch.where(atom_mask, logits, logits - inf)

    atom_alpha = torch.softmax(logits, dim=-1)  # (N, L, M, 14, 14)
    atom_alpha = mask_zero(atom_mask, atom_alpha)

    res_logits = (logits * atom_alpha).sum(dim=-1)
    res_alpha = torch.softmax(res_logits, dim=2)  # (N, L, M, 14)
    res_mask = res_mask.view(N, L, 1, 1).expand_as(res_alpha)  # (N, L, *, *)

    res_alpha = mask_zero(res_mask, res_alpha)
    return res_alpha, atom_alpha


def get_neighbors_nd_atom(a, neighbors):
    """
    a: B x L x D1 x D2 x D3 x...x Dn
    neighbors: B x L x M, M: neighbor indices
    result: B x L x M x D1 x D2 x D3 x...x Dn
    """
    L = a.shape[1]
    Ds = list(a.shape[2:])
    ND = len(Ds)
    expand_list1 = [-1, -1, L] + [-1 for _ in range(ND)]
    a = a.unsqueeze(2).expand(expand_list1)
    for _ in range(ND):
        neighbors = neighbors.unsqueeze(-1)
    expand_list2 = [-1, -1, -1] + Ds
    neighbors = neighbors.expand(expand_list2)
    result = a.gather(1, neighbors)
    return result


class GeometricEnergyAttentionAtom(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=16, query_key_dim=16, num_query_points=8,
                 num_value_points=8, num_heads=12):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.atom_node_feat_dim = node_feat_dim // 14
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points

        # Node
        self.proj_query = nn.Linear(self.atom_node_feat_dim, query_key_dim, bias=False)
        self.proj_key = nn.Linear(self.atom_node_feat_dim, query_key_dim, bias=False)
        self.proj_value = nn.Linear(self.atom_node_feat_dim, value_dim, bias=False)

        # Pair
        # self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=False)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, 14], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)

        # Output
        self.out_transform = nn.Linear(
            in_features=value_dim + (3 + 3 + 1),
            out_features=self.atom_node_feat_dim,
        )
        self.layer_norm = nn.LayerNorm(self.atom_node_feat_dim)

    def _node_logits(self, x, neighbors):
        """
        x: N, L, 14, qk_ch
        neighbors: N, L, M
        """
        query_l = self.proj_query(x)  # (N, L, 14, qk_ch)
        key_l = self.proj_key(x)  # (N, L, M, 14, qk_ch)
        key_l_knn = get_neighbors_nd_atom(key_l, neighbors)
        logits = torch.einsum('blpd,blmqd->blmpq', query_l, key_l_knn)
        return logits

    def _pos_logits(self, pos14, neighbors):
        """
        x: N, L, 14, qk_ch
        neighbors: N, L, M
        pos14: N, L, 14, 3
        """
        N, L, _, _ = pos14.size()

        q = pos14[:, :, :, None, :].expand(N, L, 14, 14, 3)
        k = pos14[:, :, None, :, :].expand(N, L, 14, 14, 3)
        k_knn = get_neighbors_nd_atom(k, neighbors)

        sum_sq_dist = ((q.unsqueeze(2) - k_knn) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logtis_beta = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / 9)) / 2)  # (N, L, M, 14, n_heads)
        return logtis_beta

    def _node_aggregation(self, res_alpha, atom_alpha, x, neighbors):
        """
        res_alpha:  Attention weights for atom.  (N, L, M, 14)
        atom_alpha:  Attention weights.  (N, L, M, 14, 14)
        x: N, L, 14, qk_ch
        neighbors: N, L, M
        """
        N, L = x.shape[:2]
        value_l = self.proj_value(x)  # (N, L, 14, qk_ch)
        value_l_knn = get_neighbors_nd_atom(value_l, neighbors)  # (N, L, M, 14, qk_ch)

        # atom aggregation
        # (N, L, M, 14, 14, 1) x (N, L, M, 1, 14, qk_ch) -> (N, L, M, 14, 14, qk_ch)
        feat_node = atom_alpha.unsqueeze(-1) * value_l_knn.unsqueeze(-3)
        feat_node = feat_node.sum(dim=-2)  # (N, L, M, 14, qk_ch)

        # residue aggregation
        # (N, L, M, 14, 1) x (N, L, M, 14, qk_ch) -> (N, L, M, 14, qk_ch)
        feat_node = res_alpha.unsqueeze(-1) * feat_node
        feat_node = feat_node.sum(dim=2)  # (N, L, 14, qk_ch)
        return feat_node

    def _pos_aggregation(self, res_alpha, atom_alpha, R, t, pos14, x, neighbors):
        """
        res_alpha:  Attention weights atom level.  (N, L, M, 14)
        atom_alpha:  Attention weights residue level.  (N, L, M, 14, 14)
        pos14:    (N, L, 14, 3).
        R:        (N, L, 14, 3, 3).
        t:        (N, L, 14, 3).
        neighbors: N, L, M
        """
        M = neighbors.shape[2]
        N, L, _, _ = pos14.size()

        v = pos14[:, :, :, None, :].expand(N, L, 14, 14, 3)
        v_knn = get_neighbors_nd_atom(v, neighbors)  # (N, L, M, 14, 14, 3).
        v_knn = pos14[:, :, None, :, None, :].expand(N, L, M, 14, 14, 3) - v_knn

        # (N, L, M, 14, 14, 1) x (N, L, M, 14, 14, 3) -> (N, L, M, 14, 14, 3)
        aggr = atom_alpha.unsqueeze(-1) * v_knn
        aggr = aggr.sum(dim=-2)  # (N, L, M, 14, 3)

        # (N, L, M, 14, 1) x (N, L, M, 14, 3) -> (N, L, M, 14, 3)
        aggr = res_alpha.unsqueeze(-1) * aggr
        aggr = aggr.sum(dim=2)  # (N, L, 14, 3)

        aggr = aggr.view(N, L * 14, 3)
        R = R.view(N, L * 14, 3, 3)
        t = t.view(N, L * 14, 3)

        feat_points = global_to_local(R, t, aggr)  # (N, L * 14, 3)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1).view(N, L, 14, 7)
        return feat_spatial

    def forward(self, R, t, pos14, x, z, atom_mask, neighbors):
        """
        Args:
            R:  Frame basis matrices, (N, L, 14, 3, 3).
            t:  Frame external (absolute) coordinates, (N, L, 14, 3).
            x:  Node-wise features, (N, L, 14, F).
            z:  Pair-wise features, (N, L, L, C).
            atom_mask:   Masks, (N, L, 14).
            neighbors:   neighbors, (N, L, M).
        Returns:
            x': Updated node-wise features, (N, L, 14, F).
        """
        # Attention logits
        logits_node = self._node_logits(x, neighbors)
        logits_spatial = self._pos_logits(pos14, neighbors)
        logits_sum = logits_node + logits_spatial

        res_alpha, atom_alpha = _alpha_from_logits_atom(logits_sum * np.sqrt(1 / 2),
                                                        atom_mask)  # (N, L, M, 14), (N, L, M, 14, 14)

        # Aggregate features
        feat_node = self._node_aggregation(res_alpha, atom_alpha, x, neighbors)
        feat_spatial = self._pos_aggregation(res_alpha, atom_alpha, R, t, pos14, x, neighbors)
        feat = torch.cat([feat_node, feat_spatial], dim=-1)

        # Finally
        feat_all = self.out_transform(feat)  # (N, L, 14, F)
        feat_all = mask_zero(atom_mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm(x + feat_all)
        return x_updated


class GEAEncoderAtom(nn.Module):
    def __init__(self, node_feat_dim, pair_feat_dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            GeometricEnergyAttentionAtom(node_feat_dim, pair_feat_dim)
            for _ in range(num_layers)
        ])

    def forward(self, R, t, p_CB, x_atom, z_atom, atom_mask, neighbors):
        for block in self.blocks:
            x = block(R, t, p_CB, x_atom, z_atom, atom_mask, neighbors)  # Residual connection within the block
        return x

class BindFormer(nn.Module):
    def __init__(self, pair_feat_dim=64, node_feat_dim=12 * 14, twotrack=False):
        super().__init__()
        self.twotrack = twotrack
        self.node_feat_dim = node_feat_dim
        if self.twotrack:
            self.res2atoms = nn.Sequential(
                nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                nn.Linear(node_feat_dim, node_feat_dim)
            )
            self.atoms2res = nn.Sequential(
                nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                nn.Linear(node_feat_dim, node_feat_dim)
            )
            self.atom_gea_layer = GEAEncoderAtom(
                node_feat_dim = node_feat_dim,
                pair_feat_dim = pair_feat_dim,
                num_layers = 1
            )
            self.atom_mlp = nn.Sequential(
                nn.Linear(node_feat_dim // 14, node_feat_dim // 14), nn.ReLU(),
                nn.Linear(node_feat_dim // 14, node_feat_dim // 14)
            )
            self.atom_norm = nn.ModuleList([nn.LayerNorm(node_feat_dim // 14) for _ in range(3)])

        self.res_gea_layer = GEAEncoderResidue(
            node_feat_dim = node_feat_dim,
            pair_feat_dim = pair_feat_dim,
            num_layers = 1
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim)
        )
        self.res_norm = nn.ModuleList([nn.LayerNorm(node_feat_dim) for _ in range(3)])

    def _update_twotrack(self, wt, mut):
        res_ga_layer = self.res_gea_layer
        res_mlp = self.res_mlp
        res_norm = self.res_norm
        atom_ga_layer = self.atom_gea_layer
        atom_mlp = self.atom_mlp
        atom_norm = self.atom_norm

        N, L = wt['complex']['aa'].shape[:2]

        wt_res_feat = wt['res']['feat']
        mut_res_feat = mut['res']['feat']
        wt_atom_feat = wt['atom']['feat']
        mut_atom_feat = mut['atom']['feat']

        wt_atom_feat = wt_atom_feat + self.res2atoms(wt_res_feat-mut_res_feat).view(N, L, 14, -1)
        mut_atom_feat = mut_atom_feat + self.res2atoms(mut_res_feat-wt_res_feat).view(N, L, 14, -1)
        wt_atom_feat = atom_norm[0](wt_atom_feat)
        mut_atom_feat = atom_norm[0](mut_atom_feat)

        wt_atom_feat = wt_atom_feat + atom_ga_layer(wt['atom']['R'], wt['atom']['t'], wt['complex']['pos_atoms'],
                                                    wt_atom_feat, wt['res']['pair_feat'],
                                                    wt['complex']['mask_atoms'], wt['complex']['neighbors'])
        mut_atom_feat = mut_atom_feat + atom_ga_layer(mut['atom']['R'], mut['atom']['t'], mut['complex']['pos_atoms'],
                                                      mut_atom_feat, mut['res']['pair_feat'],
                                                      mut['complex']['mask_atoms'], mut['complex']['neighbors'])
        wt_atom_feat = atom_norm[1](wt_atom_feat)
        mut_atom_feat = atom_norm[1](mut_atom_feat)

        wt_atom_feat = wt_atom_feat + atom_mlp(wt_atom_feat-mut_atom_feat)
        mut_atom_feat = mut_atom_feat + atom_mlp(mut_atom_feat-wt_atom_feat)
        wt_atom_feat = atom_norm[2](wt_atom_feat)
        mut_atom_feat = atom_norm[2](mut_atom_feat)

        wt_res_feat = wt_res_feat + self.atoms2res(wt_atom_feat.view(N, L, -1) - mut_atom_feat.view(N, L, -1))
        mut_res_feat = mut_res_feat + self.atoms2res(mut_atom_feat.view(N, L, -1) - wt_atom_feat.view(N, L, -1))
        wt_res_feat = res_norm[0](wt_res_feat)
        mut_res_feat = res_norm[0](mut_res_feat)
        # R, t, p_CB, x, z, mask, neighbors
        wt_res_feat = wt_res_feat + res_ga_layer(wt['res']['R'], wt['res']['t'], wt['res']['pos_cb'],
                                                 wt_res_feat, wt['res']['pair_feat'],
                                                 wt['res']['mask'], wt['complex']['neighbors'])
        mut_res_feat = mut_res_feat + res_ga_layer(mut['res']['R'], mut['res']['t'], mut['res']['pos_cb'],
                                                   mut_res_feat, mut['res']['pair_feat'],
                                                   mut['res']['mask'], mut['complex']['neighbors'])
        wt_res_feat = res_norm[1](wt_res_feat)
        mut_res_feat = res_norm[1](mut_res_feat)

        wt_res_feat = wt_res_feat + res_mlp(wt_res_feat-mut_res_feat)
        mut_res_feat = mut_res_feat + res_mlp(mut_res_feat-wt_res_feat)
        wt_res_feat = res_norm[2](wt_res_feat)
        mut_res_feat = res_norm[2](mut_res_feat)
        return wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat

    def _update(self, wt, mut):
        res_feat_wt = wt['res']['feat']
        res_feat_mut = mut['res']['feat']
        res_feat_wt = res_feat_wt + self.res_mlp(res_feat_wt-res_feat_mut)
        res_feat_mut = res_feat_mut + self.res_mlp(res_feat_mut-res_feat_wt)
        res_feat_wt = self.res_ga_layer(wt['res']['R'], wt['res']['t'], wt['res']['pos_cb'],
                                        res_feat_wt, wt['res']['pair_feat'],
                                        wt['res']['mask'], wt['complex']['neighbors'])
        res_feat_mut = self.res_ga_layer(mut['res']['R'], mut['res']['t'], mut['res']['pos_cb'],
                                         res_feat_mut, mut['res']['pair_feat'],
                                         mut['res']['mask'], mut['complex']['neighbors'])
        res_feat_wt = self.res_norm(res_feat_wt)
        res_feat_mut = self.res_norm(res_feat_mut)
        return res_feat_wt, res_feat_mut

    def forward(self, wt, mut):
        if self.twotrack:
            wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat = self._update_twotrack(wt, mut)
            return wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat
        else:
            wt_res_feat, mut_res_feat = self._update(wt, mut)
            return wt_res_feat, mut_res_feat