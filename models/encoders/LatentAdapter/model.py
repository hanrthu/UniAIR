import torch
import torch.nn as nn
from models.register import ModelRegister
from .tmp_module import MutationModelModule
from .transformer.transformer import Transformer
import torch.nn.functional as F
R = ModelRegister()
    
@R.register('vallina_transfer')
class VallinaTransfer(nn.Module):
    def __init__(
      self,
      encoder_dirs,
      latent_dim,
      hidden_dim,
      n_blocks=3,
      n_heads=16,
      fold=0,
      transfer_dirs=None,
      pre_transform_cfg={},
      **kwargs
    ):
        super().__init__()
        self.encoder = MutationModelModule.load_from_checkpoint(encoder_dirs[fold], map_location=torch.device('cpu')).model
        print("Loading checkpoint from fold {}!".format(fold))
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # for name, param in self.encoder.named_parameters():
        #     if "ddg_readout" in name:  
        #         param.requires_grad = True
            
        # if transfer_dirs is not None:
        #     print("Loading Transformer from fold {}!".format(fold))
        #     self.transformer = TransferModelModule.load_from_checkpoint(transfer_dirs[fold], map_location=torch.device('cpu')).model.transformer
        #     # for param in self.transformer.parameters():
        #     #     param.requires_grad = False
        # else:
        self.transformer = Transformer(
            embed_dim=latent_dim,
            num_heads=n_heads,
            num_blocks=n_blocks,
        )
        # self.ddg_readout = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        #     nn.Linear(latent_dim, 1)
        # )
        # self.ddg_readout = self.encoder.ddg_readout
        self.ddg_readout = self.encoder.pred_head
        # for name, param in self.ddg_readout.named_parameters():  
        #     param.requires_grad = False

    def forward(self, batch):
        batch_real = {k: v for k, v in batch['complex_wt'].items()}
        batch_pred = {k: v for k, v in batch['complex_wt'].items()}
        batch_pred['pos_atoms'] = batch_pred['pred_pos_atoms']
        batch_pred['pos_heavyatom'] = batch_pred['pred_pos_heavyatom']
        mask_residue = batch_real['mask_atoms'][:, :, 1] # CA
        
        with torch.no_grad():
            H_real = self.encoder._encode(batch_real)
            H_pred = self.encoder._encode(batch_pred)
        
        H_trans = self.transformer(
            H_pred = H_pred, 
            mask_residue = mask_residue
        )
        
        # loss_per_element_wt = F.mse_loss(H_trans, H_real, reduction='none')
        loss_dict = {}
        # loss_h_wt = (loss_per_element_wt * mask_residue.unsqueeze(-1)).sum() / mask_residue.sum()
        
        batch_mt = {k: v for k, v in batch_pred.items()}
        batch_mt_real = {k: v for k, v in batch_real.items()}
        batch_mt['aa'] = batch_mt['aa_mut']
        batch_mt_real['aa'] = batch_mt_real['aa_mut']
        with torch.no_grad():
            H_mt_real = self.encoder._encode(batch_mt_real)
            H_mt_pred = self.encoder._encode(batch_mt)
            
        H_mt_trans = self.transformer(
            H_pred = H_mt_pred, 
            mask_residue = mask_residue
        )
        # loss_per_element_mt = F.mse_loss(H_mt_trans, H_real, reduction='none')
        # loss_h_mt = (loss_per_element_mt * mask_residue.unsqueeze(-1)).sum() / mask_residue.sum()
        
        H_mt, H_wt = H_mt_trans.max(dim=1)[0], H_trans.max(dim=1)[0]
        H_mt_gt, H_wt_gt = H_mt_real.max(dim=1)[0], H_real.max(dim=1)[0]
        
        loss_h = F.mse_loss(H_mt - H_wt, H_mt_gt - H_wt_gt)
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        y = batch['labels'].squeeze()
        
        loss_ddg = F.mse_loss(ddg_pred, y)
        loss_ddg_inv = F.mse_loss(ddg_pred_inv, -y)

        loss_dict['DDG'] = (loss_ddg + loss_ddg_inv) / 2
        
        # loss_dict['H'] = (loss_h_wt + loss_h_mt) / 2
        loss_dict['H'] = loss_h
        loss_dict['X'] = 0

        return loss_dict
    
    def encode(self, batch):
        mask_residue = batch['complex_wt']['mask_atoms'][:, :, 1] # CA
        batch_pred = {k: v for k, v in batch['complex_wt'].items()}
        if 'pred_pos_atoms' in batch_pred:
            batch_pred['pos_atoms'] = batch_pred['pred_pos_atoms']
            batch_pred['pos_heavyatom'] = batch_pred['pred_pos_heavyatom']
        # self.encoder.eval()
        
        batch_wt = {k: v for k, v in batch_pred.items()}
        batch_mt = {k: v for k, v in batch_pred.items()}
        batch_mt['aa'] = batch_mt['aa_mut']


        h_wt = self.encoder._encode(batch_wt)
        h_mt = self.encoder._encode(batch_mt)
        
        h_wt = self.transformer(h_wt, mask_residue=mask_residue)
        h_mt = self.transformer(h_mt, mask_residue=mask_residue)
        
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        return H_wt, H_mt
    
    def _encode(self, data):
        mask_residue = data['mask_atoms'][:, :, 1] # CA
        h_wt = self.encoder._encode(data)
        h_wt = self.transformer(h_wt, mask_residue=mask_residue)
        return h_wt
    
    def inference(self, batch):
        mask_residue = batch['complex_wt']['mask_atoms'][:, :, 1] # CA
        batch_pred = {k: v for k, v in batch['complex_wt'].items()}
        if 'pred_pos_atoms' in batch_pred:
            batch_pred['pos_atoms'] = batch_pred['pred_pos_atoms']
            batch_pred['pos_heavyatom'] = batch_pred['pred_pos_heavyatom']
        # self.encoder.eval()
        
        batch_wt = {k: v for k, v in batch_pred.items()}
        batch_mt = {k: v for k, v in batch_pred.items()}
        batch_mt['aa'] = batch_mt['aa_mut']


        h_wt = self.encoder._encode(batch_wt)
        h_mt = self.encoder._encode(batch_mt)
        
        h_wt = self.transformer(h_wt, mask_residue=mask_residue)
        h_mt = self.transformer(h_mt, mask_residue=mask_residue)
        
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        # Mean
        # H_mt, H_wt = h_mt.mean(dim=1), h_wt.mean(dim=1)
        # Mutant
        # mut = batch['mut_flag'].long()
        # print(h_wt.shape, mut.shape, mut.sum())
        # H_mt, H_wt = (h_mt * mut.unsqueeze(-1)).sum(dim=1) / mut.unsqueeze(-1).sum(dim=1), (h_wt * mut.unsqueeze(-1)).sum(dim=1) / mut.unsqueeze(-1).sum(dim=1)
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)

        pred_dict = {
            'y_pred': ddg_pred,
            'y_pred_inv': ddg_pred_inv,
        }
        
        return pred_dict
        
class SimpleMLP(nn.Module):
    def __init__(self, latent_dim): 
        print('Initializing Simple MLP!')
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, H_pred, mask_residue=None):
        return self.mlp(H_pred)
    
@R.register('latent_adapter')
class VallinaTransferNew(nn.Module):
    def __init__(
      self,
      encoder_dirs,
      latent_dim,
      hidden_dim,
      n_blocks=3,
      n_heads=16,
      fold=0,
      transfer_dirs=None,
      encoder_type='gearbind',
      pre_transform_cfg={},
      **kwargs
    ):
        super().__init__()
        self.encoder = MutationModelModule.load_from_checkpoint(encoder_dirs[fold], map_location=torch.device('cpu')).model
        print("Loading checkpoint from fold {}!".format(fold))
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # for name, param in self.encoder.named_parameters():
        #     if "ddg_readout" in name:  
        #         param.requires_grad = True
            
        # if transfer_dirs is not None:
        #     print("Loading Transformer from fold {}!".format(fold))
        #     self.transformer = TransferModelModule.load_from_checkpoint(transfer_dirs[fold], map_location=torch.device('cpu')).model.transformer
        #     # for param in self.transformer.parameters():
        #     #     param.requires_grad = False
        # else:
        if encoder_type != 'gearbind':
            self.transformer = Transformer(
                embed_dim=latent_dim,
                num_heads=n_heads,
                num_blocks=n_blocks,
            )
        else:
            self.transformer = SimpleMLP(latent_dim)
        self.encoder_type = encoder_type
        # self.ddg_readout = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        #     nn.Linear(latent_dim, 1)
        # )
        # self.ddg_readout = self.encoder.ddg_readout
        # self.ddg_readout = self.encoder.pred_head
        # for name, param in self.ddg_readout.named_parameters():  
        #     param.requires_grad = False

    def forward(self, batch):
        batch_real = {k: v for k, v in batch.items()}
        batch_pred = {k: v for k, v in batch.items()}
        batch_pred['complex_wt']['pos_atoms'] = batch_pred['complex_wt']['pred_pos_atoms']
        batch_pred['complex_wt']['pos_heavyatom'] = batch_pred['complex_wt']['pred_pos_heavyatom']
        if 'complex_mut' in batch:
            batch_pred['complex_mut']['pos_atoms'] = batch_pred['complex_mut']['pred_pos_atoms']
            batch_pred['complex_mut']['pos_heavyatom'] = batch_pred['complex_mut']['pred_pos_heavyatom']

        mask_residue = batch_real['complex_wt']['mask_atoms'][:, :, 1] # CA
        
        
        # with torch.no_grad():
        #     H_real = self.encoder._encode(batch_real)
        #     H_pred = self.encoder._encode(batch_pred)
        with torch.no_grad():
            encoded_real = self.encoder.forward_encode(batch_real)
            encoded_pred = self.encoder.forward_encode(batch_pred)
            H_real, H_mt_real = encoded_real[0]
            H_pred, H_mt_pred = encoded_pred[0]
        
        if self.encoder_type == 'lmbind':
            mask = torch.ones(H_pred.shape[0:2]).to(H_pred.device)
            mask[:, 1:] = mask_residue
        elif self.encoder_type == 'gearbind':
            mask = None
        else:
            mask = mask_residue
        #     flatten_shape = batch_pred['complex_wt']['mask_atoms'][:, :, 1].shape
        #     mask_wt = (batch_pred['complex_wt']['mask_atoms'][:, :, 1] & batch_pred['complex_wt']['mask_atoms'][:, :, 0] & batch_pred['complex_wt']['mask_atoms'][:, :, 2]).reshape(-1)
        #     graph_size = H_pred.shape[0]
        #     expand_h_pred = torch.zeros((flatten_shape[0] * flatten_shape[1], H_pred.shape[-1]), dtype=H_pred.dtype, device=H_pred.device)
        #     indices = torch.nonzero(mask_wt, as_tuple=True)[0]
        #     # print("Expand_H:", expand_h_pred.shape, H_pred.shape)
        #     # print(indices)
        #     expand_h_pred[indices] = H_pred
        #     H_pred = expand_h_pred.reshape([flatten_shape[0], flatten_shape[1], H_pred.shape[1]])
        #     mask_mt = (batch_pred['complex_mut']['mask_atoms'][:, :, 1] & batch_pred['complex_mut']['mask_atoms'][:, :, 0] & batch_pred['complex_mut']['mask_atoms'][:, :, 2]).reshape(-1)
        #     expand_h_mt_pred = torch.zeros((flatten_shape[0]* flatten_shape[1], H_pred.shape[-1]), dtype=H_pred.dtype, device=H_pred.device)
        #     indices = torch.nonzero(mask_mt, as_tuple=True)[0]
        #     expand_h_mt_pred[indices] = H_mt_pred
        #     H_mt_pred = expand_h_mt_pred.reshape([flatten_shape[0], flatten_shape[1], H_mt_pred.shape[1]])
        #     mask = mask_wt.reshape(flatten_shape)

        H_trans = self.transformer(
            H_pred = H_pred, 
            mask_residue = mask
        ) 
        # if self.encoder_type == 'gearbind':
        #     H_trans_flatten = H_trans.reshape(-1, H_trans.shape[-1])
        #     mask_wt_flatten = mask.reshape(-1)
        #     mask_mt_flatten = mask_mt.reshape(-1)
        #     H_trans = H_trans_flatten[mask_wt_flatten]
        #     H_mt_trans_flatten = H_mt_trans.reshape(-1, H_mt_trans.shape[-1])
        #     H_mt_trans = H_mt_trans_flatten[mask_mt_flatten]
        # loss_per_element_mt = F.mse_loss(H_mt_trans, H_real, reduction='none')
        # loss_h_mt = (loss_per_element_mt * mask_residue.unsqueeze(-1)).sum() / mask_residue.sum()
        if self.encoder_type != 'ppiformer':
            H_mt_trans = self.transformer(
            H_pred = H_mt_pred, 
            mask_residue = mask
        )
            _, H_real_odds = self.encoder.forward_readout((H_real, H_mt_real), *encoded_real[1:])
            pred_dict_trans, H_trans_odds = self.encoder.forward_readout((H_trans, H_mt_trans), *encoded_pred[1:])
        else:
            _, H_real_odds = self.encoder.forward_readout((H_real, None), *encoded_real[1:])
            pred_dict_trans, H_trans_odds = self.encoder.forward_readout((H_trans, None), *encoded_pred[1:])
        
        loss_h = F.mse_loss(H_trans_odds, H_real_odds)
        ddg_pred = pred_dict_trans['y_pred']
        y = batch['labels'].squeeze()
        loss_ddg = F.mse_loss(ddg_pred, y)
        if 'y_pred_inv' in pred_dict_trans:
            ddg_pred_inv = pred_dict_trans['y_pred_inv']
            loss_ddg_inv = F.mse_loss(ddg_pred_inv, -y)
            loss_ddg = (loss_ddg + loss_ddg_inv) / 2
        
        loss_dict = {
            'DDG': loss_ddg,
            'H': loss_h,
            'X': 0
        }

        return loss_dict
    
    
    def inference(self, batch):
        mask_residue = batch['complex_wt']['mask_atoms'][:, :, 1] # CA

        batch_pred = {k: v for k, v in batch.items()}
        if 'pred_pos_atoms' in batch_pred:
            batch_pred['complex_wt']['pos_atoms'] = batch_pred['complex_wt']['pred_pos_atoms']
            batch_pred['complex_wt']['pos_heavyatom'] = batch_pred['complex_wt']['pred_pos_heavyatom']
        # self.encoder.eval()
        
        encoded_pred = self.encoder.forward_encode(batch_pred)
        h_wt, h_mt = encoded_pred[0]
        if self.encoder_type == 'lmbind':
            mask = torch.ones(h_wt.shape[0:2]).to(h_wt.device)
            mask[:, 1:] = mask_residue
        elif self.encoder_type == 'gearbind':
            mask = None
        else:
            mask = mask_residue
            # flatten_shape = batch_pred['complex_wt']['mask_atoms'][:, :, 1].shape
            # mask_wt = (batch_pred['complex_wt']['mask_atoms'][:, :, 1] & batch_pred['complex_wt']['mask_atoms'][:, :, 0] & batch_pred['complex_wt']['mask_atoms'][:, :, 2]).reshape(-1)
            # graph_size = h_wt.shape[0]
            # expand_h_pred = torch.zeros((flatten_shape[0] * flatten_shape[1], h_wt.shape[-1]), dtype=h_wt.dtype, device=h_wt.device)
            # indices = torch.nonzero(mask_wt, as_tuple=True)[0]
            # # print("Expand_H:", expand_h_pred.shape, H_pred.shape)
            # # print(indices)
            # expand_h_pred[indices] = h_wt
            # h_wt = expand_h_pred.reshape([flatten_shape[0], flatten_shape[1], h_wt.shape[1]])
            # mask_mt = (batch_pred['complex_mut']['mask_atoms'][:, :, 1] & batch_pred['complex_mut']['mask_atoms'][:, :, 0] & batch_pred['complex_mut']['mask_atoms'][:, :, 2]).reshape(-1)
            # expand_h_mt_pred = torch.zeros((flatten_shape[0]* flatten_shape[1], h_mt.shape[-1]), dtype=h_mt.dtype, device=h_mt.device)
            # indices = torch.nonzero(mask_mt, as_tuple=True)[0]
            # expand_h_mt_pred[indices] = h_mt
            # h_mt = expand_h_mt_pred.reshape([flatten_shape[0], flatten_shape[1], h_mt.shape[1]])
            # mask = mask_wt.reshape(flatten_shape)

        h_wt = self.transformer(h_wt, mask_residue=mask)
        if self.encoder_type != 'ppiformer':
            h_mt = self.transformer(h_mt, mask_residue=mask)
        # if self.encoder_type == 'gearbind':
        #     H_trans_flatten = h_wt.reshape(-1, h_wt.shape[-1])
        #     mask_wt_flatten = mask.reshape(-1)
        #     mask_mt_flatten = mask_mt.reshape(-1)
        #     h_wt = H_trans_flatten[mask_wt_flatten]
        #     H_mt_trans_flatten = h_mt.reshape(-1, h_mt.shape[-1])
        #     h_mt = H_mt_trans_flatten[mask_mt_flatten]
        
            pred_dict, _ = self.encoder.forward_readout((h_wt, h_mt), *encoded_pred[1:])
        else:
            pred_dict, _ = self.encoder.forward_readout((h_wt, None), *encoded_pred[1:])
 
        return pred_dict
        