from ._base import register_transform
from typing import Dict

@register_transform('pre_rde_transform')
class PreRDETransform(object):
    def __init__(
        self,
        resolution: str = 'backbone+CB',
    ):
        super().__init__()
        self.resolution = resolution

    def __call__(self, data: Dict) -> tuple:
        if self.resolution == 'backbone':
            data['pos_atoms'] = data['pos_atoms'][:, :, :4]
            data['mask_atoms'] = data['mask_atoms'][:, :, :4]
            data['bfactor_atoms'] = data['bfactor_atoms'][:, :, :4]
        elif self.resolution == 'backbone+CB':
            data['pos_atoms'] = data['pos_atoms'][:, :, :5]
            data['mask_atoms'] = data['mask_atoms'][:, :, :5]
            data['bfactor_atoms'] = data['bfactor_atoms'][:, :, :5]
        return data