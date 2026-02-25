# This is the implementation of UniPPI, including UniAlign
import torch.nn as nn
from typing import Dict
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from models.register import ModelRegister
from models.encoders.RDE.encoders.single import PerResidueEncoder
from models.encoders.RDE.encoders.pair import ResiduePairEncoder
from models.encoders.RDE.encoders.attn import GAEncoder
from data.transforms.rde import PreRDETransform
R = ModelRegister()

def get_model(model_args:dict=None, output_dim=1, fold=None):
    register = ModelRegister()
    if fold != None:
        model_args_ori = {'output_dim': output_dim, 'fold': fold}
    else:
        model_args_ori = {'output_dim': output_dim}
    model_args_ori.update(model_args)
    model_cls = register[model_args['model_type']]
    model = model_cls(**model_args_ori)
    return model

def parse_yaml(yaml_dir):
    with open(yaml_dir, 'r') as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
        # args = Namespace(**config_dict)
    return config_dict

@R.register("uniair")
class UniAIR(nn.Module):
    def __init__(self, model_dirs, encoder, pre_transform_cfg= {'resolution': 'backbone+CB'}, fold=0, fix_backbone=True, **kwargs):
        super().__init__()
        from pl_modules import MutationModelModule
        self.pretrained_models = nn.ModuleList([])
        self.model_names = model_dirs.keys()
        config_dict = {'gearbind': 1024, 'rde': 128, 'ppiformer': 128, 'essm': 320}
        embeds = [] 
        for config_name in model_dirs:
            model = MutationModelModule.load_from_checkpoint(model_dirs[config_name][fold], map_location=torch.device('cpu')).model
            print("Loading checkpoint from fold {}!".format(fold))
            self.pretrained_models.append(model)
            embeds.append(config_dict[config_name])
            if fix_backbone:
                for p in model.parameters():
                    p.requires_grad_(False)
        num_models = len(self.pretrained_models)
        
        dim = encoder.node_feat_dim
        self.transform = PreRDETransform(**pre_transform_cfg)
        self.single_encoder = PerResidueEncoder(
            feat_dim=encoder.node_feat_dim,
            max_num_atoms=5,  # N, CA, C, O, CB,
        )
        
        
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=encoder.pair_feat_dim,
            max_num_atoms=5,  # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(**encoder)
        
        self.weights = nn.Sequential(
            nn.Linear(sum(embeds) + 128, dim), nn.ReLU(),
            nn.Linear(dim, num_models)
        )

    def gate(self, data):
        batch = self.transform(data)
        if batch['aa'].shape[0] == 0:
            return batch['aa'].unsqueeze(-1).repeat(1, 1, self.dim)
        mask_residue = batch['mask_atoms'][:, :, 1]
        if 'mut_flag' in batch:
            mut_flag = batch['mut_flag']
        else:
            mut_flag = torch.zeros(batch['aa'].shape, device=batch['aa'].device)
        chi = batch['chi'] * (1 - mut_flag.float())[:, :, None]
        x = self.single_encoder(
            aa=batch['aa'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue,
        )
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

    def inference(self, batch: Dict):
        data1 = {k: v for k, v in batch['complex_wt'].items()}
        data2 = {k: v for k, v in batch['complex_wt'].items()}
        data2['aa'] = data2['aa_mut']
        x_wt, x_mt = self.gate(data1).max(dim=1)[0], self.gate(data2).max(dim=1)[0]
        H_ipa = x_wt - x_mt
        if len(H_ipa.shape) == 1:
            H_ipa = H_ipa.unsqueeze(0)        
        embeddings = []
        preds = []
        for i, (model, encoder_type) in enumerate(zip(self.pretrained_models, self.model_names)):
            h_in = model.forward_encode(batch)
            pred_dict, _ = model.forward_readout(h_in)

            # if len(h_in.shape) == 1:
            #     h_in = h_in.unsqueeze(0)    
            if encoder_type == 'gearbind':
                embeddings.append(h_in['ba'].squeeze())
            else:
                embeddings.append(h_in.squeeze())  
            pred = pred_dict['y_pred'].squeeze().unsqueeze(-1) 
            preds.append(pred)
        # print(embeddings)
        x = torch.cat([H_ipa] + embeddings, dim=-1)
        temperature = 2.5
        weights = F.softmax(self.weights(x) / temperature, dim=1)
        ddg_pred = torch.sum(torch.hstack(preds) * weights, dim=1, keepdim=True)
        pred_dict = {
            'y_pred': ddg_pred,
        }
        return pred_dict

@R.register("uniair_meanens")
class UniPPI_mean(nn.Module):
    def __init__(self, model_dirs, fold=0, fix_backbone=True, **kwargs):
        super().__init__()
        from pl_modules import MutationModelModule
        self.pretrained_models = nn.ModuleList([])
        self.model_names = model_dirs.keys()
        for config_name in model_dirs:
            model = MutationModelModule.load_from_checkpoint(model_dirs[config_name][fold], map_location=torch.device('cpu')).model
            print("Loading checkpoint from fold {}!".format(fold))
            self.pretrained_models.append(model)
            if fix_backbone:
                for p in model.parameters():
                    p.requires_grad_(False)
        
    def inference(self, batch: Dict):
        preds = []
        for i, model in enumerate(self.pretrained_models):
            pred = model.inference(batch)['y_pred'].squeeze().unsqueeze(-1)
            preds.append(pred)
            # print(i, pred.shape)
        ddg_pred = torch.mean(torch.hstack(preds), dim=1, keepdim=True)
        pred_dict = {
            'y_pred': ddg_pred,
        }
        return pred_dict

