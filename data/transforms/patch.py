import random
import torch

from ._base import _index_select_data, register_transform, _get_CB_positions


@register_transform('focused_random_patch')
class FocusedRandomPatch(object):

    def __init__(self, focus_attr, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.focus_attr = focus_attr
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        focus_flag = (data[self.focus_attr] > 0)  # (L, )
        if focus_flag.sum() == 0:
            # If there is no active residues, randomly pick one.
            focus_flag[random.randint(0, focus_flag.size(0) - 1)] = True
        seed_idx = torch.multinomial(focus_flag.float(), num_samples=1).item()

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx + 1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]  # (L, 1) -> (L, )
        nbh_seed_idx = dist_from_seed.argsort()[:self.seed_nbh_size]  # (Nb, )

        core_idx = nbh_seed_idx[focus_flag[nbh_seed_idx]]  # (Ac, ), the core-set must be a subset of the focus-set
        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]  # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]  # (P, ) # 在Coreset里面的元素离自己最近的距离是0，所以一定会被选上。
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('random_patch')
class RandomPatch(object):

    def __init__(self, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        seed_idx = random.randint(0, data['aa'].size(0) - 1)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx + 1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]  # (L, 1) -> (L, )
        core_idx = dist_from_seed.argsort()[:self.seed_nbh_size]  # (Nb, )

        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]  # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]  # (P, )
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_region_with_padding_patch')
class SelectedRegionWithPaddingPatch(object):

    def __init__(self, select_attr, each_residue_nbh_size, patch_size_limit):
        super().__init__()
        self.select_attr = select_attr
        self.each_residue_nbh_size = each_residue_nbh_size
        self.patch_size_limit = patch_size_limit

    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        pos_sel = pos_CB[select_flag]  # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel)  # (L, S)
        nbh_sel_idx = torch.argsort(dist_from_sel, dim=0)[:self.each_residue_nbh_size, :]  # (nbh, S)
        patch_idx = nbh_sel_idx.view(-1).unique()  # (patchsize,)

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_region_fixed_size_patch')
class SelectedRegionFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size

    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)
        try:
            pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
            pos_sel = pos_CB[select_flag]  # (S, 3)
        except:
            print("Flag missing or shape not match!")
        # print("Pos CB and sel:", pos_sel.shape, pos_CB.shape, select_flag.shape)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]  # (L, )
        # print(self.patch_size)
        patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]
        patch_idx, _ = torch.sort(patch_idx)

        data_patch = _index_select_data(data, patch_idx)
        data_patch['patch_idx'] = patch_idx
        return data_patch
    
@register_transform('interface_fixed_size_patch')
class InterfaceFixedSizePatch(object):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, data):
        
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        # print("Pos CB and sel:", pos_sel.shape, pos_CB.shape, select_flag.shape)
        the_same_chain = (data['group_id'].unsqueeze(1) == data['group_id'].unsqueeze(0)).int() * 10086
        # print(the_same_chain[0])
        dist_matrix = torch.cdist(pos_CB, pos_CB)  # (L, )
        # print(dist_matrix[0])
        dist_matrix = dist_matrix + the_same_chain
        min_dist = dist_matrix.min(dim=1)[0]
        # print(self.patch_size)
        # patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]
        patch_idx = torch.argsort(min_dist)[:self.patch_size]
        patch_idx, _ = torch.sort(patch_idx)
        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('interface_patch')
class SelectedInterfacePatch(object):

    def __init__(self, interface_dist=6, select_attr='mut_flag'):
        super().__init__()
        self.interface_dist = interface_dist
        self.select_attr = select_attr

    def __call__(self, data):
        
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        # print("Pos CB and sel:", pos_sel.shape, pos_CB.shape, select_flag.shape)
        the_same_chain = (data['group_id'].unsqueeze(1) == data['group_id'].unsqueeze(0)).int() * 10086
        # print(the_same_chain[0])
        dist_matrix = torch.cdist(pos_CB, pos_CB)  # (L, )
        # print(dist_matrix[0])
        dist_matrix = dist_matrix + the_same_chain
        min_dist = dist_matrix.min(dim=1)[0]
        select_flag = (data[self.select_attr] > 0)
        selected_idx = torch.nonzero(select_flag, as_tuple=True)[0]
        # print(self.patch_size)
        # patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]
        patch_idx = torch.nonzero(min_dist < self.interface_dist, as_tuple=True)[0]
        patch_idx, _ = torch.sort(torch.unique(torch.cat((patch_idx, selected_idx))))
        data_patch = _index_select_data(data, patch_idx)
        return data_patch
    
