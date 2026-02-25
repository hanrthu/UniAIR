import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class DDGAttention(nn.Module):

    def __init__(self, input_dim, output_dim, value_dim=16, query_key_dim=16, num_heads=12):
        super(DDGAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_heads = num_heads

        self.query = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.key   = nn.Linear(input_dim, query_key_dim*num_heads, bias=False)
        self.value = nn.Linear(input_dim, value_dim*num_heads, bias=False)

        self.out_transform = nn.Linear(
            in_features = (num_heads*value_dim) + (num_heads*(3+3+1)),
            out_features = output_dim,
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def _alpha_from_logits(self, logits, mask, inf=1e5):
        """
        Args:
            logits: Logit matrices, (N, L_i, L_j, num_heads).
            mask:   Masks, (N, L).
        Returns:
            alpha:  Attention weights.
        """
        N, L, _, _ = logits.size()
        mask_row = mask.view(N, L, 1, 1).expand_as(logits)      # (N, L, *, *)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)     # (N, L, L, *)
        
        logits = torch.where(mask_pair, logits, logits-inf)
        alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        """
        Args:
            x:  (..., num_heads * num_channels)
        Returns:
            (..., num_heads, num_channels)
        """
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, x, pos_CA, pos_CB, frame, mask):
        # Attention logits
        query = self._heads(self.query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, head_size)
        key = self._heads(self.key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, head_size)
        logits_node = torch.einsum('blhd, bkhd->blkh', query, key)
        alpha = self._alpha_from_logits(logits_node, mask)  # (N, L, L, n_heads)

        value = self._heads(self.value(x), self.num_heads, self.value_dim)  # (N, L, n_heads, head_size)
        feat_node = torch.einsum('blkh, bkhd->blhd', alpha, value).flatten(-2)
        
        rel_pos = pos_CB.unsqueeze(1) - pos_CA.unsqueeze(2)  # (N, L, L, 3)
        atom_pos_bias = torch.einsum('blkh, blkd->blhd', alpha, rel_pos)  # (N, L, n_heads, 3)
        feat_distance = atom_pos_bias.norm(dim=-1)
        feat_points = torch.einsum('blij, blhj->blhi', frame, atom_pos_bias)  # (N, L, n_heads, 3)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        feat_spatial = torch.cat([
            feat_points.flatten(-2), 
            feat_distance, 
            feat_direction.flatten(-2),
        ], dim=-1)

        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)

        feat_all = self.out_transform(feat_all)  # (N, L, F)
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        if x.shape[-1] == feat_all.shape[-1]:
            x_updated = self.layer_norm(x + feat_all)
        else:
            x_updated = self.layer_norm(feat_all)

        return x_updated