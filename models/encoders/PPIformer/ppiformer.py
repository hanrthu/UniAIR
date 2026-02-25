from typing import Dict
import torch
import torch.nn as nn
from typing import Literal
from data.transforms.ppiformer import PreEquiformerTransform
from models.register import ModelRegister
from equiformer_pytorch import Equiformer
from torch_geometric.nn import MLP
R = ModelRegister()

# Implementation of PPIformer with wt_marginals strategy
@R.register('ppiformer')
class PPIformer(nn.Module):
    def __init__(self, encoder_checkpoint, classifier_checkpoint, pre_transform_cfg, output_dim=1, **kwargs):
        super().__init__()
        self.encoder = Equiformer(
            dim=(128, 64),
            dim_in=(20, 1),
            num_degrees=2,
            input_degrees=2,
            heads=2,
            dim_head=(32, 16),
            depth=8,
            valid_radius=8,  # 32 / 4.0 (EquiFold value / RFdiffusion coord normalization)
            num_neighbors=10,
            num_edge_tokens=2,
            edge_dim=16,
            gate_attn_head_outputs=False,
        )
        self.classifier = MLP(in_channels=128, hidden_channels=128, out_channels=20, num_layers=1)
        self.encoder.load_state_dict(torch.load(encoder_checkpoint))
        self.classifier.load_state_dict(torch.load(classifier_checkpoint))
        torch.load(classifier_checkpoint, map_location='cpu')
        
        self.corrector = torch.nn.Linear(1, output_dim)
        self.transform = PreEquiformerTransform(**pre_transform_cfg)
    

    def forward_encode(self, batch):
        data = batch['complex_wt']
        inputs = self.transform(data)
        wt = torch.clamp(data['aa'], min=0, max=19)
        mt = torch.clamp(data['aa_mut'], min=0, max=19)
        mut_flag = data['mut_flag']
        out = self.encoder(**inputs)
        h = out.type0 * (inputs['mask'].unsqueeze(-1))
        y_logits = self.classifier(h)
        y_proba = y_logits.softmax(dim=-1)
        wt_proba = torch.gather(y_proba, 2, wt.unsqueeze(-1))
        mt_proba = torch.gather(y_proba, 2, mt.unsqueeze(-1))
        log_odds = wt_proba.log() - mt_proba.log()
        embedding = (log_odds * mut_flag.unsqueeze(-1))
        return embedding
    
    def forward_readout(self, h_in):
        ddg_pred = h_in.sum(dim=1)
        ddg_pred = self.corrector(ddg_pred)
        pred_dict = {
            'y_pred': ddg_pred.squeeze()
        }
        return pred_dict, h_in.sum(dim=1)
    
    def forward(self, batch: Dict) -> torch.Tensor:
        output = self.forward_encode(batch)
        pred_dict, _ = self.forward_readout(output)
        return pred_dict
    
    def inference(self, batch):
        return self.forward(batch)  
        
        
if __name__ == '__main__':
    a = PPIformer()