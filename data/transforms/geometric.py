import random
import torch
import numpy as np

from ._base import register_transform

@register_transform('subtract_center_of_mass')
class SubtractCOM(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        pos = data['pos_atoms']
        mask = data['mask_atoms']
        if mask is None:
            center = torch.zeros(3, device=pos.device)
        elif mask.sum() == 0:
            center = torch.zeros(3, device=pos.device)
        else:
            center = pos[mask].mean(axis=0)
        data['pos_atoms'] = pos - center[None, None, :]
        if 'pred_pos_atoms' in data:
            pred_pos = data['pred_pos_atoms']
            if mask is None:
                pred_center = torch.zeros(3, device=pos.device)
            elif mask.sum() == 0:
                pred_center = torch.zeros(3, device=pos.device)
            else:
                pred_center = pred_pos[mask].mean(axis=0)
            data['pred_pos_atoms'] = pred_pos - pred_center[None, None, :]
        return data
    
