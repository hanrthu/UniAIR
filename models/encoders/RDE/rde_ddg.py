import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.single import PerResidueEncoder
from .encoders.pair import ResiduePairEncoder
from .encoders.attn import GAEncoder
from .rde import CircularSplineRotamerDensityEstimator
from data.transforms.rde import PreRDETransform
from models.register import ModelRegister
R = ModelRegister()

@R.register('rde_ddg')
class DDG_RDE_Network(nn.Module):
    def __init__(self, rde_checkpoint, encoder, pre_transform_cfg= {'resolution': 'backbone+CB'}, output_dim=1, rde_grad=False, **kwargs):
        super().__init__()

        # Pretrain
        ckpt = torch.load(rde_checkpoint, map_location='cpu')
        self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
        self.rde.load_state_dict(ckpt['model'])
        # print("Successfully Loaded Model!")
        if rde_grad is False:
            print("Fixing RDE parameters")
            for p in self.rde.parameters():
                p.requires_grad_(False)
        dim = ckpt['config'].model.encoder.node_feat_dim
        self.dim = dim
        self.transform = PreRDETransform(**pre_transform_cfg)
        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=encoder.node_feat_dim,
            max_num_atoms=5,  # N, CA, C, O, CB,
        )
        self.single_fusion = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mut_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=dim,
            padding_idx=0,
        )

        self.pair_encoder = ResiduePairEncoder(
            feat_dim=encoder.pair_feat_dim,
            max_num_atoms=5,  # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(**encoder)

        # Pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, output_dim)
        )

    def _encode_rde(self, batch, mask_extra=None):
        batch = self.transform(batch)
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        if 'mut_flag' in batch:
            mut_flag = batch['mut_flag']
        else:
            mut_flag = torch.zeros(batch['aa'].shape, device=batch['aa'].device)
        batch['chi_masked_flag'] = mut_flag
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        # with torch.no_grad():
        return self.rde.encode(batch)

    def _encode(self, data):
        batch = self.transform(data)
        if batch['aa'].shape[0] == 0:
            return batch['aa'].unsqueeze(-1).repeat(1, 1, self.dim)
        mask_residue = batch['mask_atoms'][:, :, 1] # CA
        if 'mut_flag' in batch:
            mut_flag = batch['mut_flag']
        else:
            mut_flag = torch.zeros(batch['aa'].shape, device=batch['aa'].device)
        chi = batch['chi'] * (1 - mut_flag.float())[:, :, None]

        x_single = self.single_encoder(
            aa=batch['aa'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue,
        )
        x_pret = self._encode_rde(batch)
        x = self.single_fusion(torch.cat([x_single, x_pret], dim=-1))
        b = self.mut_bias(mut_flag.long())
        x = x + b

        z = self.pair_encoder(
            aa=batch['aa'],
            res_nb=batch['res_nb'], chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'],
        )

        x = self.attn_encoder(
            pos_atoms=batch['pos_atoms'],
            res_feat=x, pair_feat=z,
            mask=mask_residue
        )

        return x

    def forward_encode(self, batch):
        batch_wt = {k: v for k, v in batch['complex_wt'].items()}
        batch_mt = {k: v for k, v in batch['complex_wt'].items()}
        batch_mt['aa'] = batch_mt['aa_mut']
        h_wt = self._encode(batch_wt)
        h_mt = self._encode(batch_mt)
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        embedding = H_mt - H_wt
        return embedding
    
    def forward_readout(self, h_in):
        # Original: Max pooling
        ddg_pred = self.ddg_readout(h_in).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(-h_in).squeeze(-1)
        pred_dict = {
            'y_pred': ddg_pred,
            'y_pred_inv': ddg_pred_inv,
        }
        return pred_dict, h_in
    
    def forward(self, batch):
        output = self.forward_encode(batch)
        pred_dict, _ = self.forward_readout(output)
        return pred_dict
    
    def inference(self, batch):
        return self.forward(batch)



@R.register('rde_ddg_lite')
class DDG_RDE_Lite(nn.Module):

    def __init__(self, rde_checkpoint, encoder, output_dim=1, **kwargs):
        super().__init__()

        # Pretrain
        ckpt = torch.load(rde_checkpoint, map_location='cpu')
        self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
        self.rde.load_state_dict(ckpt['model'])
        # print("Successfully Loaded Model!")
        for p in self.rde.parameters():
            p.requires_grad_(False)
        dim = ckpt['config'].model.encoder.node_feat_dim
        self.dim = dim
        self.mut_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=dim,
            padding_idx=0,
        )
        # Pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, output_dim)
        )

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch):
        if batch['aa'].shape[0] == 0:
            return batch['aa'].unsqueeze(-1).repeat(1, 1, self.dim)
        x = self._encode_rde(batch)
        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b
        return x

    def forward(self, batch):

        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        
        # Try 1: mean pooling
        # H_mt, H_wt = h_mt.mean(dim=1), h_wt.mean(dim=1)
        # Try 2: mutation pooling
        # mut = batch['mut_flag'].long()
        # H_mt, H_wt = h_mt[mut], h_wt[mut]

        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        pred_dict = {
            'y_pred': ddg_pred,
            'y_pred_inv': ddg_pred_inv,
        }
        return pred_dict


@R.register('rde_ddg_scratch')
class DDG_RDE_Scratch(nn.Module):

    def __init__(self, rde_checkpoint, encoder, output_dim=1, **kwargs):
        super().__init__()

        # Pretrain
        ckpt = torch.load(rde_checkpoint, map_location='cpu')
        # print("Successfully Loaded Model!")
        dim = ckpt['config'].model.encoder.node_feat_dim
        self.dim = dim
        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=encoder.node_feat_dim,
            max_num_atoms=5,  # N, CA, C, O, CB,
        )
        # self.single_fusion = nn.Sequential(
        #     nn.Linear(2 * dim, dim), nn.ReLU(),
        #     nn.Linear(dim, dim)
        # )
        self.mut_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=dim,
            padding_idx=0,
        )

        self.pair_encoder = ResiduePairEncoder(
            feat_dim=encoder.pair_feat_dim,
            max_num_atoms=5,  # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(**encoder)

        # Pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, output_dim)
        )


    def encode(self, batch):
        if batch['aa'].shape[0] == 0:
            return batch['aa'].unsqueeze(-1).repeat(1, 1, self.dim)
        mask_residue = batch['mask_atoms'][:, :, 1] # CA
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x_single = self.single_encoder(
            aa=batch['aa'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue,
        )
        x = x_single
        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b

        z = self.pair_encoder(
            aa=batch['aa'],
            res_nb=batch['res_nb'], chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'],
        )

        x = self.attn_encoder(
            pos_atoms=batch['pos_atoms'],
            res_feat=x, pair_feat=z,
            mask=mask_residue
        )

        return x

    def forward(self, batch):

        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        pred_dict = {
            'y_pred': ddg_pred,
            'y_pred_inv': ddg_pred_inv,
        }
        return pred_dict


