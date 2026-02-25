import torch
import torch.nn as nn

from models.encoders.RDE.common.geometry import construct_3d_basis, global_to_local

class PositionalEncoding(nn.Module):
    
    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs-1, num_funcs))
    
    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code

class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.aatype_embed = nn.Embedding(21, feat_dim)
        self.torsion_embed = PositionalEncoding()
        self.mlp = nn.Sequential(
            nn.Linear(21*14*3 + feat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, pos14, atom_mask):
        """
        Args:
            aa:           (N, L).
            pos14:        (N, L, 14, 3).
            atom_mask:    (N, L, 14).
        """
        N, L = aa.size()

        R = construct_3d_basis(pos14[:, :, 1], pos14[:, :, 2], pos14[:, :, 0])  # (N, L, 3, 3)
        t = pos14[:, :, 1]  # (N, L, 3)
        crd14 = global_to_local(R, t, pos14)    # (N, L, 14, 3)
        crd14_mask = atom_mask[:, :, :, None].expand_as(crd14)
        crd14 = torch.where(crd14_mask, crd14, torch.zeros_like(crd14))

        aa_expand  = aa[:, :, None, None, None].expand(N, L, 21, 14, 3)
        rng_expand = torch.arange(0, 21)[None, None, :, None, None].expand(N, L, 21, 14, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd14[:, :, None, :, :].expand(N, L, 21, 14, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, 21 * 14 * 3)

        # print("AA extreme", torch.max(aa), torch.min(aa))
        aa_feat = self.aatype_embed(aa) # (N, L, feat)
        out_feat = self.mlp(torch.cat([crd_feat, aa_feat], dim=-1))
        return out_feat