import torch
from torch import nn

from models.encoders.LatentAdapter.transformer.modules import TransformerEncoder
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, use_flash_attn=False):
        super().__init__()
        self.encoder = TransformerEncoder(embed_dim, 
                                          num_blocks,
                                          num_heads,
                                          use_rot_emb=True,
                                          attn_qkv_bias=False,
                                          transition_dropout=0.0, 
                                          attention_dropout=0.0, 
                                          residual_dropout=0.0, 
                                          transition_factor=4,
                                          use_flash_attn=use_flash_attn)
        self.use_flash_attn = use_flash_attn

    def forward(self, H_pred, sample_sequence=True, sample_structure=True, mask_residue=None, need_attn_weights=False):

        if mask_residue is not None:
            pad_mask = torch.logical_not(mask_residue)
        else:
            pad_mask = None
        H_trans, attn_weights = self.encoder(H_pred, key_padding_mask=pad_mask, need_attn_weights=need_attn_weights)

        return H_trans

    # @torch.no_grad()
    # def sample(self, H_pred, X_pre, mask_residue=None, need_attn_weights=False):
    #     pad_mask = torch.logical_not(mask_residue)
    #     H_trans, attn_weights = self.encoder(H_pred, key_padding_mask=pad_mask, need_attn_weights=need_attn_weights)
    #     return H_trans
