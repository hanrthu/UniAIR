import dataclasses
import io
import copy
import random

import torch
from torch import cdist
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import pickle

from pathlib import Path
from sklearn.metrics import pairwise_distances

from data.protein import proteins
from data.protein.proteins import ProteinInput
from Bio.PDB.Polypeptide import one_to_index
from tqdm import tqdm
from data.pdbredo_dataset import _process_structure
from data.transforms import get_transform, _get_CB_positions
from data.register import DataRegister
from pymol import cmd
from chempy import Atom, models
from data.protein.residue_constants import lexico_restypes

R = DataRegister()
ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def get_knn_dis_index_mat(pos14, mask14, n_neighbor=None):
    """
    pos14: [L, 14, 3] float
    mask14: [L, 14, 3] float
    inter_res_index: [L, n_neighbor] int
    """
    if n_neighbor == None:
        return torch.arange(pos14.shape[0])
    cb_pos = _get_CB_positions(pos14, mask14)
    # ca_pos = pos14[:, ATOM_CA]

    # dis_mat = pairwise_distances(cb_pos, cb_pos)
    dis_mat = torch.cdist(cb_pos, cb_pos)
    dis_mat_zero_mask = dis_mat == 0
    dis_mat[dis_mat_zero_mask] = torch.inf
    dis_min_arg = dis_mat.argsort(dim=-1)
    inter_res_index = dis_min_arg[:, :n_neighbor]
    return inter_res_index

def create_object_from_numpy(object_name, coords):
    """
    Creates a PyMOL object from a NumPy array of coordinates.

    Parameters:
    object_name (str): The name of the object in PyMOL.
    coords (np.ndarray): NumPy array of shape (N, 3) with [x, y, z] coordinates.
    """
    model = models.Indexed()
    for i, (x, y, z) in enumerate(coords):
        atom = Atom()
        atom.coord = [x, y, z]
        atom.name = f"C{i+1}"
        atom.resn = "DUM"
        atom.resi = "1"
        atom.chain = "A"
        model.add_atom(atom)
    cmd.load_model(model, object_name)

@R.register('with_mutation_dataset')
class WithMutationDataset(Dataset):
    '''
    The implementation of Protein Mutation Dataset
    '''
    def __init__(self, dataframe, data_root, max_length=256,
                 col_wt='path_wt', col_mut='path_mut', col_mutation='mutation', col_chainids='chainids',
                 col_valid_chains='valid_chains',
                 cols_label=['DDG'],
                 col_partner='Partners(A_B)',
                 col_pred=None,
                 col_protein=None,
                 col_pdb=None,
                 n_neighbors=None,
                 diskcache=None,
                 transform=None,
                 **kwargs):
        # self.data_loader = ComplexLoader(data_root)
        self.data_root = data_root
        self.df: pd.DataFrame = dataframe.copy()
        if col_chainids not in self.df:
            self.df[col_chainids] = ''
        self.df[col_chainids] = self.df[col_chainids].fillna('').astype('str')

        self.max_length = max_length
        self.n_neighbors = n_neighbors

        self.col_wt = col_wt
        self.col_mut = col_mut
        self.col_protein = col_protein
        self.col_partner = col_partner
        self.col_mutation = col_mutation
        self.col_chainids = col_chainids
        self.valid_chains = col_valid_chains
        self.cols_label = cols_label
        self.col_pdb = col_pdb
        self.col_pred = col_pred
        
        self.diskcache = diskcache
        self.transform = get_transform(transform)
        # for col in self.cols_label:
        #     self.df[col] = self.df.get(col, np.nan)
        self._load_entries()
        self.structures = {}

    def __len__(self):
        return len(self.df)
    
    def _load_entries(self):
        self.entries = []
        def _parse_mut(mut_name):
            wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
            mutseq = int(mut_name[2:-1])
            return {
                'wt': wt_type,
                'mt': mt_type,
                'chain': mutchain,
                'resseq': mutseq,
                'icode': ' ',
                'name': mut_name
            }
        for i, row in self.df.iterrows():
            # print("Row:", row)
            complex = row[self.col_protein]
            mut_str = row[self.col_mutation]
            muts = list(map(_parse_mut, row[self.col_mutation].split(',')))
            pdbcode = row[self.col_pdb][:-4]
            group_ligand = row[self.col_partner].split('_')[0]
            group_receptor = row[self.col_partner].split('_')[1]
            labels = row[self.cols_label]
            pdb_path = row[self.col_wt]
            mut_path = row[self.col_mut]
            if row[self.valid_chains] is not None:
                valid_chains = row[self.valid_chains].split(',')
            else:
                valid_chains = None
            chainids = row[self.col_chainids]
            entry = {
                'id': i,
                'complex': complex,
                'mutstr': mut_str,
                'num_muts': len(muts),
                'pdbcode': pdbcode,
                'group_ligand': list(group_ligand),
                'group_receptor': list(group_receptor),
                'mutations': muts,
                'labels': labels,
                'valid_chains': valid_chains,
                'chainids': chainids,
                'path_wt': pdb_path,
                'path_mut': mut_path
            }
            if self.col_pred is not None:
                pred_pdb_path = row[self.col_pred]
                entry['pred_path_wt'] = pred_pdb_path
                entry['pred_path_mut'] = pred_pdb_path.replace('clean', 'mt/skempi_v2').replace('.pdb', '_' + mut_str.replace(',', '_')+'.pdb')
            self.entries.append(entry)

    def process_structure(self, path_wt, path_mut, valid_chains, path_pred_wt=None, path_pred_mut=None):
        pdbcode_wt = path_wt.split('/')[-1][:-4]
        pdbcode_mut = path_mut.split('/')[-1][:-4]
        if (pdbcode_wt, pdbcode_mut) in self.structures:
            return self.structures[(pdbcode_wt, pdbcode_mut)]
        if self.diskcache is None or (pdbcode_wt, pdbcode_mut) not in self.diskcache:
            data_wt = _process_structure(path_wt, structure_id=pdbcode_wt, valid_chains=valid_chains)
            data_mut = _process_structure(path_mut, structure_id=pdbcode_mut, valid_chains=valid_chains)
            if path_pred_wt is not None:
                # print("Path pred wt is not None!")
                data_pred_wt = _process_structure(path_pred_wt, structure_id=pdbcode_wt+'_pred', valid_chains=valid_chains)
                data_pred_mut = _process_structure(path_pred_mut, structure_id=pdbcode_mut+'_pred', valid_chains=valid_chains)
                data_wt['pred_pos_heavyatom'] = data_pred_wt['pos_heavyatom']
                data_mut['pred_pos_heavyatom'] = data_pred_mut['pos_heavyatom']
            seq_map = {}
            for j, (chain_id, resseq, icode) in enumerate(zip(data_wt.chain_id, data_wt.resseq, data_wt.icode)):
                seq_map[(chain_id, int(resseq), icode)] = j
            self.structures[(pdbcode_wt, pdbcode_mut)] = (data_wt, data_mut, seq_map)
            if self.diskcache is not None:
                self.diskcache[(pdbcode_wt, pdbcode_mut)] = (data_wt, data_mut, seq_map)
        else:
            self.structures[(pdbcode_wt, pdbcode_mut)] = self.diskcache[(pdbcode_wt, pdbcode_mut)]
        return self.structures[(pdbcode_wt, pdbcode_mut)]
    
    def __len__(self):
        return len(self.df)
    
    def get_data(self, idx):
        # We need some columns for the dataframe
        entry = self.entries[idx]
        valid_chains = self.entries[idx]['valid_chains']
        path_wt = entry['path_wt']
        path_mut = entry['path_mut']
        # pdbcode_wt = path_wt.split('/')[-1][:-4]
        # pdbcode_mut = path_mut.split('/')[-1][:-4]
        if 'pred_path_wt' not in entry:
            data_wt, data_mut, seq_map = copy.deepcopy(self.process_structure(path_wt, path_mut, valid_chains))
        else:
            data_wt, data_mut, seq_map = copy.deepcopy(self.process_structure(path_wt, path_mut, valid_chains, entry['pred_path_wt'], entry['pred_path_mut']))
        
        labels = pd.to_numeric(np.array(entry['labels']))
        labels_mask = np.logical_not(np.isnan(labels))
        labels = np.nan_to_num(labels, nan=np.random.randn() * 0.001)
        
        group_id = []
        for ch in data_wt['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        
        #  Update chains to a binary form in Unibind
        # if isinstance(chainids, str) and (chainids != ''):
        #     indices = [chains.index(chainid) if chainid in chains else -1 for chainid in chainids]
        #     indices = [i for i in indices if i != -1]
        #     chainid_wt_new = 1 - np.isin(p14_wt.chain_id, indices).astype('int')
        #     p14_wt = dataclasses.replace(p14_wt, chain_id=chainid_wt_new)
        #     p14_mut = dataclasses.replace(p14_mut, chain_id=chainid_wt_new)
        data_wt['group_id'] = torch.LongTensor(group_id)
        data_mut['group_id'] = torch.LongTensor(group_id)
        
        aa_mut = data_wt['aa'].clone()
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
            if ch_rs_ic not in seq_map: continue
            aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            
        data_wt['aa_mut'] = aa_mut
        
        # Flag is the same
        data_wt['mut_flag'] = (data_wt['aa'] != data_wt['aa_mut'])
        data_mut['mut_flag'] = (data_wt['aa'] != data_wt['aa_mut'])

        seq_list = []
        mut_seq_list = []
        aa = data_wt['aa']
        aa_mut = data_wt['aa_mut']
        for chain_nb in data_wt['chain_nb'].unique():
            seq = ''.join(lexico_restypes[i] for i in aa[data_wt['chain_nb'] == chain_nb])
            mut_seq = ''.join(lexico_restypes[i] for i in aa_mut[data_wt['chain_nb'] == chain_nb])
            assert len(seq) == len(mut_seq)
            seq_list.append(seq)
            mut_seq_list.append(mut_seq)
        prot_lengths = [len(seq) for seq in seq_list]
        max_prot_length = max(prot_lengths)
        
        if self.transform is not None:
            data_wt = self.transform(data_wt)
            data_mut = self.transform(data_mut)
        
        data_wt['neighbors'] = get_knn_dis_index_mat(data_wt['pos_atoms'], data_wt['mask_atoms'], self.n_neighbors)
        data_mut['neighbors'] = get_knn_dis_index_mat(data_mut['pos_atoms'], data_mut['mask_atoms'], self.n_neighbors)
        
        if self.col_pred is not None:
            pos_pred = data_wt['pred_pos_atoms'].numpy()[:, 1, :].squeeze()
            pos_real = data_wt['pos_atoms'].numpy()[:, 1, :].squeeze()
            cmd.reinitialize()
            create_object_from_numpy('pos_pred', pos_pred)
            create_object_from_numpy('pos_real', pos_real)
            _ = cmd.align("pos_pred", "pos_real")
            matrix = cmd.get_object_matrix("pos_pred")
            m = np.array(matrix).reshape((4,4))
            rotation_matrix = m[:3, :3]
            translation_vector = m[:3, 3]
            aligned_coords = np.einsum('lbi,ij->lbj', data_wt['pred_pos_atoms'], rotation_matrix.T) + translation_vector
            aligned_coords = torch.FloatTensor(aligned_coords)
            data_wt['pred_pos_atoms'] = aligned_coords
            
            pos_pred = data_mut['pred_pos_atoms'].numpy()[:, 1, :].squeeze()
            pos_real = data_mut['pos_atoms'].numpy()[:, 1, :].squeeze()
            cmd.reinitialize()
            create_object_from_numpy('pos_pred', pos_pred)
            create_object_from_numpy('pos_real', pos_real)
            _ = cmd.align("pos_pred", "pos_real")
            matrix = cmd.get_object_matrix("pos_pred")
            m = np.array(matrix).reshape((4,4))
            rotation_matrix = m[:3, :3]
            translation_vector = m[:3, 3]
            aligned_coords = np.einsum('lbi,ij->lbj', data_mut['pred_pos_atoms'], rotation_matrix.T) + translation_vector
            aligned_coords = torch.FloatTensor(aligned_coords)
            data_mut['pred_pos_atoms'] = aligned_coords
        
        data_wt.pop('chain_id')
        data_wt.pop('icode')
        data_mut.pop('chain_id')
        data_mut.pop('icode')
        data = {
            'complex': entry['complex'],
            'mutstr': entry['mutstr'],
            'complex_wt': data_wt,
            'complex_mut': data_mut,
            'labels': torch.FloatTensor(labels),
            'label_mask': torch.BoolTensor(labels_mask),
            'group_id': torch.LongTensor(group_id),
            'prot_seqs': seq_list,
            'mut_seqs': mut_seq_list,
            'max_prot_length': max_prot_length,
        }
        return data

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            case = self.df.iloc[idx]
            paths_wt = case[self.col_wt].split(',')
            paths_mut = case[self.col_mut].split(',')
            labels = case[self.cols_label]
            print("Error: ", case, paths_wt, paths_mut, labels)
            raise e

@R.register('wo_mutation_dataset')
class WoMutationDataset(Dataset):
    def __init__(self, dataframe, data_root, max_length=256, col_wt='path_wt', col_mut='path_mut', col_mutation='mutation',
                 n_neighbors=None, col_pdb='pdb_id', col_protein='Protein', col_pred=None, col_partner='Partners(A_B)', cols_label=['DDG'],
                 transform=None, diskcache=None, col_valid_chains='valid_chains', **kwargs):
        self.df: pd.DataFrame = dataframe.copy()
        self.data_root = data_root
        self.max_length = max_length
        self.n_neighbors = n_neighbors

        self.col_wt = col_wt
        self.col_mut = col_mut
        self.col_protein = col_protein
        self.col_partner = col_partner
        self.col_mutation = col_mutation
        self.cols_label = cols_label
        self.col_pdb = col_pdb
        self.valid_chains = col_valid_chains
        self.diskcache = diskcache
        self.transform = get_transform(transform)
        self.diskcache = diskcache
        self.structures = None
        if col_pred is not None:
            self.col_pred = col_pred
        else:
            self.col_pred = None
        self._load_entries()
        self._load_structures(reset=False)

    def _load_entries(self):
        self.entries = []
        def _parse_mut(mut_name):
            wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
            mutseq = int(mut_name[2:-1])
            return {
                'wt': wt_type,
                'mt': mt_type,
                'chain': mutchain,
                'resseq': mutseq,
                'icode': ' ',
                'name': mut_name
            }

        for i, row in self.df.iterrows():
            complex = row[self.col_protein]
            mut_str = row[self.col_mutation]
            muts = list(map(_parse_mut, row[self.col_mutation].split(',')))
            pdbcode = row[self.col_pdb][:-4]
            group_ligand = row[self.col_partner].split('_')[0]
            group_receptor = row[self.col_partner].split('_')[1]
            labels = row[self.cols_label]
            if row[self.valid_chains] is not None:
                valid_chains = row[self.valid_chains].split(',')
            else:
                valid_chains = None
            pdb_path = row[self.col_wt]
            entry = {
                'id': i,
                'complex': complex,
                'mutstr': mut_str,
                'num_muts': len(muts),
                'pdbcode': pdbcode,
                'group_ligand': list(group_ligand),
                'group_receptor': list(group_receptor),
                'mutations': muts,
                'labels': pd.to_numeric(labels),
                'valid_chains': valid_chains,
                'pdb_path': pdb_path,
            }
            if self.col_pred is not None:
                pred_pdb_path = row[self.col_pred]
                entry['pred_pdb_path'] = pred_pdb_path
            self.entries.append(entry)

    def _preprocess_structures(self):
        structures = {}
        pdb_paths = list(set([e['pdb_path'] for e in self.entries]))
        # pdb_paths = [e['pdb_path'] for e in self.entries]
        # valid_chains = [e['valid_chains'] for e in self.entries]
        if self.col_pred is not None:
            paired_paths = set([(e['pdb_path'], e['pred_pdb_path']) for e in self.entries])
            pdb_paths = [item[0] for item in paired_paths]
            pred_pdb_paths = [item[1] for item in paired_paths]
            
        for i, pdb_path in enumerate(tqdm(pdb_paths, desc='Structures')):
            # pdb_path = os.path.join(self.data_root, 'PDBs', '{}.pdb'.format(pdbcode))
            pdbcode = pdb_path.split('/')[-1][:-4]
            # print("PDB Code:", pdbcode)
            if self.diskcache is None or pdbcode not in self.diskcache:
                data = _process_structure(pdb_path, pdbcode)
                if self.col_pred is not None:
                    pred_pdb_path = pred_pdb_paths[i]
                    data_pred = _process_structure(pred_pdb_path, pdbcode+'_pred')
                    data['pred_pos_heavyatom'] = data_pred['pos_heavyatom']
                seq_map = {}
                for j, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
                    seq_map[(chain_id, resseq.item(), icode)] = j
                structures[pdbcode] = (data, seq_map)
                if self.diskcache is not None:
                    self.diskcache[pdbcode] = (data, seq_map)
            else:
                structures[pdbcode] = self.diskcache[pdbcode]
        return structures

    def _load_structures(self, reset):
        self.structures = self._preprocess_structures()

    def __len__(self):
        return len(self.df)

    def get_data(self, idx):
        entry = self.entries[idx]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode'}
        for k in keys:
            data[k] = entry[k]

        labels = pd.to_numeric(np.array(entry['labels']))
        labels_mask = np.logical_not(np.isnan(labels))
        labels = np.nan_to_num(labels, nan=np.random.randn() * 0.001)

        data['labels'] = torch.FloatTensor(labels)
        data['label_mask'] = torch.BoolTensor(labels_mask)
        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)

        aa_mut = data['aa'].clone()
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
            if ch_rs_ic not in seq_map: 
                print("Mutation not found:", entry['complex'], ch_rs_ic)
                continue
            aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])
        
        seq_list = []
        mut_seq_list = []
        aa = data['aa']
        aa_mut = data['aa_mut']
        for chain_nb in data['chain_nb'].unique():
            seq = ''.join(lexico_restypes[i] for i in aa[data['chain_nb'] == chain_nb])
            mut_seq = ''.join(lexico_restypes[i] for i in aa_mut[data['chain_nb'] == chain_nb])
            assert len(seq) == len(mut_seq)
            seq_list.append(seq)
            mut_seq_list.append(mut_seq)
        prot_lengths = [len(seq) for seq in seq_list]
        max_prot_length = max(prot_lengths)
        
        if self.transform is not None:
            # try:
            data = self.transform(data) 
            # except:
            #     print("The data is abnormal:", data['complex'], data['mutstr'])
                
        if self.col_pred is not None:
            pos_pred = data['pred_pos_atoms'].numpy()[:, 1, :].squeeze()
            pos_real = data['pos_atoms'].numpy()[:, 1, :].squeeze()
            cmd.reinitialize()
            create_object_from_numpy('pos_pred', pos_pred)
            create_object_from_numpy('pos_real', pos_real)
            _ = cmd.align("pos_pred", "pos_real")
            matrix = cmd.get_object_matrix("pos_pred")
            m = np.array(matrix).reshape((4,4))
            rotation_matrix = m[:3, :3]
            translation_vector = m[:3, 3]
            aligned_coords = np.einsum('lbi,ij->lbj', data['pred_pos_atoms'], rotation_matrix.T) + translation_vector
            aligned_coords = torch.FloatTensor(aligned_coords)
            data['pred_pos_atoms'] = aligned_coords
        
        # valid_mask = torch.ones_like(data['ddG']) - torch.isnan(data['ddG'])
        data.pop('chain_id')
        data.pop('icode')
        # data['max_prot_length'] = max_prot_length
        # data['prot_seqs'] = seq_list
        # data['mut_seqs'] = mut_seq_list
        data = {
            'complex': entry['complex'],
            'complex_wt': data,
            'labels': torch.FloatTensor(labels),
            'label_mask': torch.BoolTensor(labels_mask),
            'mutstr': entry['mutstr'],
            'group_id': torch.LongTensor(group_id),
            'prot_seqs': seq_list,
            'mut_seqs': mut_seq_list,
            'max_prot_length': max_prot_length,
        }
        # print("Data:", data)
        return data

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            case = self.df.iloc[idx]
            paths_wt = case[self.col_wt].split(',')
            paths_mut = case[self.col_mut].split(',')
            labels = case[self.cols_label]
            print("Error: ", case, paths_wt, paths_mut, labels)
            raise e
        
@R.register('unialign_dataset')
class UniAlignDataset(Dataset):
    def __init__(self, dataframe, data_root, max_length=256, col_struct='path',
                 n_neighbors=None, col_pdb='pdb_id', col_protein='Protein', col_pred=None, col_partner='Partners(A_B)', cols_label=['DDG'],
                 transform=None, diskcache=None, col_valid_chains='valid_chains', **kwargs):
        self.df: pd.DataFrame = dataframe.copy()
        self.data_root = data_root
        self.max_length = max_length
        self.n_neighbors = n_neighbors

        self.col_struct = col_struct
        self.col_protein = col_protein
        self.col_partner = col_partner
        self.cols_label = cols_label
        self.col_pdb = col_pdb
        self.valid_chains = col_valid_chains
        self.diskcache = diskcache
        self.transform = get_transform(transform)
        self.diskcache = diskcache
        self.structures = None
        if col_pred is not None:
            self.col_pred = col_pred
        else:
            self.col_pred = None
        self._load_entries()
        # self._load_structures(reset=False)
        self.structures = {}

    def _load_entries(self):
        self.entries = []
        for i, row in self.df.iterrows():
            complex = row[self.col_protein]
            pdbcode = row[self.col_pdb][:-4]
            group_ligand = row[self.col_partner].split('_')[0]
            group_receptor = row[self.col_partner].split('_')[1]
            labels = row[self.cols_label]
            if row[self.valid_chains] is not None:
                valid_chains = row[self.valid_chains].split(',')
            else:
                valid_chains = None
            pdb_path = row[self.col_struct]
            entry = {
                'id': i,
                'complex': complex,
                'pdbcode': pdbcode,
                'group_ligand': list(group_ligand),
                'group_receptor': list(group_receptor),
                'labels': pd.to_numeric(labels),
                'valid_chains': valid_chains,
                'pdb_path': pdb_path,
            }
            if self.col_pred is not None:
                pred_pdb_path = row[self.col_pred]
                entry['pred_pdb_path'] = pred_pdb_path
            self.entries.append(entry)

    def process_structure(self, pdb_path, valid_chains, pred_pdb_path=None):
        pdbcode = pdb_path.split('/')[-1][:-4]
        if pdbcode in self.structures:
            return self.structures[pdbcode]
        if self.diskcache is None or pdbcode not in self.diskcache:
            data = _process_structure(pdb_path, pdbcode, valid_chains)
            if pred_pdb_path is not None:
                data_pred = _process_structure(pred_pdb_path, pdbcode+'_pred', valid_chains[i])
                data['pred_pos_heavyatom'] = data_pred['pos_heavyatom']
            self.structures[pdbcode] = data
            if self.diskcache is not None:
                self.diskcache[pdbcode] = data
        else:
            self.structures[pdbcode] = self.diskcache[pdbcode]
        return self.structures[pdbcode]
    
    # def _preprocess_structures(self):
    #     structures = {}
    #     pdb_paths = list(set([e['pdb_path'] for e in self.entries]))
    #     valid_chains = [e['valid_chains'] for e in self.entries]
    #     if self.col_pred is not None:
    #         paired_paths = set([(e['pdb_path'], e['pred_pdb_path']) for e in self.entries])
    #         pdb_paths = [item[0] for item in paired_paths]
    #         pred_pdb_paths = [item[1] for item in paired_paths]
            
    #     for i, pdb_path in enumerate(tqdm(pdb_paths, desc='Structures')):
    #         # pdb_path = os.path.join(self.data_root, 'PDBs', '{}.pdb'.format(pdbcode))
    #         pdbcode = pdb_path.split('/')[-1][:-4]
    #         # print("PDB Code:", pdbcode)
    #         if self.diskcache is None or pdbcode not in self.diskcache:
    #             data = _process_structure(pdb_path, pdbcode, valid_chains[i])
    #             if self.col_pred is not None:
    #                 pred_pdb_path = pred_pdb_paths[i]
    #                 data_pred = _process_structure(pred_pdb_path, pdbcode+'_pred', valid_chains[i])
    #                 data['pred_pos_heavyatom'] = data_pred['pos_heavyatom']
    #             structures[pdbcode] = data
    #             if self.diskcache is not None:
    #                 self.diskcache[pdbcode] = data
    #         else:
    #             structures[pdbcode] = self.diskcache[pdbcode]
    #     return structures

    # def _load_structures(self, reset):
    #     self.structures = self._preprocess_structures()

    def __len__(self):
        return len(self.df)

    def get_data(self, idx):
        entry = self.entries[idx]
        valid_chains = self.entries[idx]['valid_chains']
        if self.col_pred is not None:
            pred_pdb_path = self.entries[idx]['pred_pdb_path']
        else:
            pred_pdb_path = None
        pdb_path = self.entries[idx]['pdb_path']
        data = copy.deepcopy(self.process_structure(pdb_path, valid_chains, pred_pdb_path))
        keys = {'id', 'complex', 'pdbcode'}
        for k in keys:
            data[k] = entry[k]

        labels = pd.to_numeric(np.array(entry['labels']))
        labels_mask = np.logical_not(np.isnan(labels))
        labels = np.nan_to_num(labels, nan=np.random.randn() * 0.001)

        data['labels'] = torch.FloatTensor(labels)
        data['label_mask'] = torch.BoolTensor(labels_mask)
        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)
        
        seq_list = []
        aa = data['aa']
        for chain_nb in data['chain_nb'].unique():
            seq = ''.join(lexico_restypes[i] for i in aa[data['chain_nb'] == chain_nb])
            seq_list.append(seq)
        prot_lengths = [len(seq) for seq in seq_list]
        max_prot_length = max(prot_lengths)
        
        if self.transform is not None:
            try:
                data = self.transform(data)
            except:
                print("The data is abnormal:", data['complex'], data['mutstr'])
                
        if self.col_pred is not None:
            pos_pred = data['pred_pos_atoms'].numpy()[:, 1, :].squeeze()
            pos_real = data['pos_atoms'].numpy()[:, 1, :].squeeze()
            cmd.reinitialize()
            create_object_from_numpy('pos_pred', pos_pred)
            create_object_from_numpy('pos_real', pos_real)
            _ = cmd.align("pos_pred", "pos_real")
            matrix = cmd.get_object_matrix("pos_pred")
            m = np.array(matrix).reshape((4,4))
            rotation_matrix = m[:3, :3]
            translation_vector = m[:3, 3]
            aligned_coords = np.einsum('lbi,ij->lbj', data['pred_pos_atoms'], rotation_matrix.T) + translation_vector
            aligned_coords = torch.FloatTensor(aligned_coords)
            data['pred_pos_atoms'] = aligned_coords
        
        # valid_mask = torch.ones_like(data['ddG']) - torch.isnan(data['ddG'])
        data.pop('chain_id')
        data.pop('icode')
        data = {
            'complex': entry['complex'],
            'complex_wt': data,
            'labels': torch.FloatTensor(labels),
            'label_mask': torch.BoolTensor(labels_mask),
            'group_id': torch.LongTensor(group_id),
            'prot_seqs': seq_list,
            'mutstr': '',
            'max_prot_length': max_prot_length
        }
        return data

    def __getitem__(self, idx):
        i = 0
        while i < 10:
            try:
                return self.get_data(idx)
            except Exception as e:
                case = self.df.iloc[idx]
                paths_wt = case[self.col_struct].split(',')
                labels = case[self.cols_label]
                print("Error Processing: ", case, paths_wt, labels, "Trying another idx...")
                i += 1
                idx = random.randint(0, len(self.df))
                # raise e
