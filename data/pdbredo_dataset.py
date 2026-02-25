import os
import math
import random
import pickle
import collections
import torch
import lmdb
import numpy as np
from easydict import EasyDict
from typing import Mapping, List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
import math
from torch.utils.data._utils.collate import default_collate
from data.transforms import get_transform
from data.register import DataRegister
import esm
R = DataRegister()

import data.protein.proteins as proteins

ClusterIdType = str
PdbCodeType = str
ChainIdType = str

def _process_structure(structure_path, structure_id, valid_chains=None) -> Optional[Dict]:

    protein = proteins.ProteinInput.from_path(structure_path, with_angles=True, return_dict=True, valid_chains=valid_chains)
    if protein is None:
        print(f'[INFO] Failed to parse structure. Too few valid residues: {structure_path}')
        return None
    chains = list(protein.keys())
    # print("Protein_Path:", structure_path)
    # breakpoint()
    protein: proteins.ProteinInput = proteins.proteins_merge([protein[chain] for chain in chains],
                                                              np.arange(len(chains)))
    atom37 = protein.atom_positions
    mask37 = protein.atom_mask
    protein = protein.to_atom14()
    data = EasyDict({
        'chain_id': protein.chain_id, 'chain_nb': torch.LongTensor(protein.chain_nb),
        'resseq': torch.LongTensor(protein.resseq), 'icode': protein.icode, 'res_nb': torch.LongTensor(protein.res_nb),
        'aa': torch.LongTensor(protein.aatype_lexico),
        'pos_heavyatom': torch.FloatTensor(protein.atom_positions), 'mask_heavyatom': torch.BoolTensor(protein.atom_mask),
        'bfactor_heavyatom': torch.FloatTensor(protein.b_factors),
        'phi': torch.FloatTensor(protein.torsion_angles[:, 1]), 'phi_mask': torch.BoolTensor(protein.torsion_angles_mask[:, 1]),
        'psi': torch.FloatTensor(protein.torsion_angles[:, 2]), 'psi_mask': torch.BoolTensor(protein.torsion_angles_mask[:, 2]),
        'chi': torch.FloatTensor(protein.torsion_angles[:, 3:]), 'chi_alt': torch.FloatTensor(protein.alt_torsion_angles[:, 3:]),
        'chi_mask': torch.BoolTensor(protein.torsion_angles_mask[:, 3:]), 'chi_complete': torch.BoolTensor(np.ones(protein.torsion_angles.shape[0])),
        'atom_pos37': torch.FloatTensor(atom37), 'atom_mask37': torch.FloatTensor(mask37)
    })
    data['id'] = structure_id
    return data

@R.register('pdbredo_chain_dataset')
class PDBRedoChainDataset(Dataset):
    MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB

    def __init__(
            self,
            split,
            pdbredo_dir='./data/PDB_REDO',
            clusters_path='./data/pdbredo_clusters.txt',
            splits_path='./data/pdbredo_splits.txt',
            processed_dir='./data/PDB_REDO_processed',
            num_preprocess_jobs=math.floor(cpu_count() * 0.8),
            transform=None,
            reset=False,
    ):
        super().__init__()
        self.pdbredo_dir = pdbredo_dir
        self.clusters_path = clusters_path
        self.splits_path = splits_path
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        self.num_preprocess_jobs = num_preprocess_jobs
        self.transform = get_transform(transform)

        self.clusters: Mapping[
            ClusterIdType, List[Tuple[PdbCodeType, ChainIdType]]
        ] = collections.defaultdict(list)
        self.splits: Mapping[
            str, List[ClusterIdType]
        ] = collections.defaultdict(list)
        self._load_clusters()
        self._load_splits()

        # Structure cache
        self.db_conn = None
        self.db_keys: Optional[List[PdbCodeType]] = None
        self._preprocess_structures(reset)

        # Sanitize clusters
        self._sanitize_clusters(reset)

        # Select clusters of the split
        self._clusters_of_split = [
            c
            for c in self.splits[split]
            if c in self.clusters
        ]

    @property
    def lmdb_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')

    @property
    def keys_path(self):
        return os.path.join(self.processed_dir, 'keys.pkl')

    @property
    def sanitized_clusters_path(self):
        return os.path.join(self.processed_dir, 'sanitized_clusters.pkl')

    def _load_clusters(self):
        with open(self.clusters_path, 'r') as f:
            lines = f.readlines()
        current_cluster = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                if word[0] == '[' and word[-1] == ']':
                    current_cluster = word[1:-1]
                else:
                    pdbcode, chain_id = word.split(':')
                    self.clusters[current_cluster].append((pdbcode, chain_id))

    def _load_splits(self):
        with open(self.splits_path, 'r') as f:
            lines = f.readlines()
        current_split = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                if word[0] == '[' and word[-1] == ']':
                    current_split = word[1:-1]
                else:
                    self.splits[current_split].append(word)

    def get_all_pdbcodes(self):
        pdbcodes = set()
        for _, pdbchain_list in self.clusters.items():
            for pdbcode, _ in pdbchain_list:
                pdbcodes.add(pdbcode)
        return pdbcodes

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path) and not reset:
            return
        pdbcodes = self.get_all_pdbcodes()
        tasks = []
        for pdbcode in pdbcodes:
            cif_path = os.path.join(
                self.pdbredo_dir, pdbcode[1:3], pdbcode, f"{pdbcode}_final.cif"
            )
            if not os.path.exists(cif_path):
                print(f'[WARNING] CIF not found: {cif_path}.')
                continue
            tasks.append(
                delayed(_process_structure)(cif_path, pdbcode)
            )

        # Split data into chunks
        chunk_size = 8192
        task_chunks = [
            tasks[i * chunk_size:(i + 1) * chunk_size]
            for i in range(math.ceil(len(tasks) / chunk_size))
        ]

        # Establish database connection
        db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )

        keys = []
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(
                    task
                    for task in tqdm(task_chunk, desc=f"Chunk {i + 1}/{len(task_chunks)}")
                )
                stored = 0
                for data in processed:
                    if data is None:
                        continue
                    key = data['id']
                    keys.append(key)
                    txn.put(key=key.encode(), value=pickle.dumps(data))
                    stored += 1
                print(f"[INFO] {stored} processed for chunk#{i + 1}")
        db_conn.close()

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)

    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)

    def _close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_keys = None

    def _get_from_db(self, pdbcode):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(pdbcode.encode()))  # Made a copy
        return data

    def _sanitize_clusters(self, reset):
        if os.path.exists(self.sanitized_clusters_path) and not reset:
            with open(self.sanitized_clusters_path, 'rb') as f:
                self.clusters = pickle.load(f)
            return

        # Step 1: Find structures and chains that do not exist in PDB_REDO
        clusters_raw = self.clusters
        pdbcode_to_chains: Dict[PdbCodeType, List[ChainIdType]] = collections.defaultdict(list)
        for pdbcode, pdbchain_list in clusters_raw.items():
            for pdbcode, chain in pdbchain_list:
                pdbcode_to_chains[pdbcode].append(chain)

        pdb_removed, chain_removed = 0, 0
        pdbcode_to_chains_ok = {}
        self._connect_db()
        for pdbcode, chain_list in tqdm(pdbcode_to_chains.items(), desc='Sanitize'):
            if pdbcode not in self.db_keys:
                pdb_removed += 1
                continue
            data = self._get_from_db(pdbcode)
            ch_exists = []
            for ch in chain_list:
                if ch in data['chain_id']:
                    ch_exists.append(ch)
                else:
                    chain_removed += 1
            if len(ch_exists) > 0:
                pdbcode_to_chains_ok[pdbcode] = ch_exists
            else:
                pdb_removed += 1

        print(f'[INFO] Structures removed: {pdb_removed}. Chains removed: {chain_removed}.')
        pdbchains_allowed = set(
            (p, c)
            for p, clist in pdbcode_to_chains_ok.items()
            for c in clist
        )

        # Step 2: Rebuild the clusters according to the allowed chains.
        pdbchain_to_clust = {}
        for clust_name, pdbchain_list in clusters_raw.items():
            for pdbchain in pdbchain_list:
                if pdbchain in pdbchains_allowed:
                    pdbchain_to_clust[pdbchain] = clust_name

        clusters_sanitized = collections.defaultdict(list)
        for pdbchain, clust_name in pdbchain_to_clust.items():
            clusters_sanitized[clust_name].append(pdbchain)

        print('[INFO] %d clusters after sanitization (from %d).' % (len(clusters_sanitized), len(clusters_raw)))

        with open(self.sanitized_clusters_path, 'wb') as f:
            pickle.dump(clusters_sanitized, f)
        self.clusters = clusters_sanitized

    def __len__(self):
        return len(self._clusters_of_split)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index, None)

        # Select cluster
        clust = self._clusters_of_split[index[0]]
        pdbchain_list = self.clusters[clust]

        # Select a pdb-chain from the cluster and retrieve the data point
        if index[1] is None:
            pdbcode, chain = random.choice(pdbchain_list)
        else:
            pdbcode, chain = pdbchain_list[index[1]]
        data = self._get_from_db(pdbcode)  # Made a copy

        # Focus on the chain
        focus_flag = torch.zeros(len(data['chain_id']), dtype=torch.bool)
        for i, ch in enumerate(data['chain_id']):
            if ch == chain: focus_flag[i] = True
        data['focus_flag'] = focus_flag
        data['focus_chain'] = chain

        if self.transform is not None:
            data = self.transform(data)

        return data


def get_pdbredo_chain_dataset(cfg):
    from .transforms import get_transform
    return PDBRedoChainDataset(
        split=cfg.split,
        pdbredo_dir=cfg.pdbredo_dir,
        clusters_path=cfg.clusters_path,
        splits_path=cfg.splits_path,
        processed_dir=cfg.processed_dir,
        transform=get_transform(cfg.transform),
    )


DEFAULT_PAD_VALUES = {
    'aa': 20,
    'aa_masked': 20,
    'mask_atoms': 0,
    'aa_true': 20,
    'chain_nb': -1,
    'pos14': 0.0,
    'chain_id': ' ',
    'icode': ' ',
}

EXCLUDE_KEYS = ['labels', 'label_mask']

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, exclude_keys=EXCLUDE_KEYS, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.exclude_keys = exclude_keys
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            # assert x.size(0) <= n
            if x.size(0) >= n:
                return x[:n]
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def collate_complex(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys_inter = self._get_common_keys(data_list)
        keys = []
        keys_not_pad = []
        keys_ignore = ['prot_seqs', 'mut_seqs', 'max_prot_length']
        for key in keys_inter:
            if key in keys_ignore:
                continue
            elif key not in self.exclude_keys:
                keys.append(key)
            else:
                keys_not_pad.append(key)

        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            for k in keys_not_pad:
                data_padded[k] = data[k]
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return data_list_padded

    def pad_for_esm(self, batch):
        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        mut_flag = 0
        prot_chains = [len(item['prot_seqs']) for item in batch]
        
        max_item_prot_length = [item['max_prot_length'] for item in batch]
        max_prot_length = max(max_item_prot_length)
        total_prot_chains = sum(prot_chains)
        if self.eight:
            max_prot_length = math.ceil((max_prot_length + 2) / 8) * 8
        else:
            max_prot_length = max_prot_length + 2
        prot_batch = torch.empty([total_prot_chains, max_prot_length])
        prot_batch.fill_(prot_alphabet.padding_idx)
        if 'mut_seqs' in batch[0]:
            mut_flag = 1
            mut_batch = torch.empty([total_prot_chains, max_prot_length])
            mut_batch.fill_(prot_alphabet.padding_idx)
        curr_prot_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            if 'mut_seqs' in item:
                mut_seqs = item['mut_seqs']
            for i, prot_seq in enumerate(prot_seqs):
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                if 'mut_seqs' in item:
                    mut_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                    mut_seq_encode = prot_alphabet.encode(mut_seqs[i])
                    seq_m = torch.tensor(mut_seq_encode, dtype=torch.int64)
                    mut_batch[curr_prot_idx, 1: len(mut_seq_encode)+1] = seq_m
                    mut_batch[curr_prot_idx, len(mut_seq_encode)+1] = prot_alphabet.eos_idx
                curr_prot_idx += 1
        prot_mask = torch.zeros_like(prot_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        if mut_flag:
            return prot_batch.long(), mut_batch.long(), prot_chains, prot_mask
        else:
            return prot_batch.long(), prot_chains, prot_mask

    def __call__(self, data_list):
        if 'complex_wt' in data_list[0] and 'complex_mut' in data_list[0]:
            data_list_padded = []
            complex_wts = [data['complex_wt'] for data in data_list]
            complex_wt_list_padded = self.collate_complex(complex_wts)
            # batch_wt = default_collate(complex_wt_list_padded)
            
            complex_muts = [data['complex_mut'] for data in data_list]
            complex_mut_list_padded = self.collate_complex(complex_muts)
            
            label_list = [{'labels': data['labels'], 'label_mask': data['label_mask'], 'complex': data['complex'], 'mutstr': data['mutstr']} for data in data_list]
            # batch_mut = default_collate(complex_mut_list_padded)
            for wt, mut, label_info in zip(complex_wt_list_padded, complex_mut_list_padded, label_list):
                data_list_padded.append({'complex': label_info['complex'], 'mutstr': label_info['mutstr'], 'complex_wt': wt, 'complex_mut': mut, 'labels': label_info['labels'], 'label_mask': label_info['label_mask']})
        elif 'complex_wt' in data_list[0]:
            data_list_padded = []
            complex_wts = [data['complex_wt'] for data in data_list]
            complex_wt_list_padded = self.collate_complex(complex_wts)
            label_list = [{'labels': data['labels'], 'label_mask': data['label_mask'], 'complex': data['complex'], 'mutstr': data['mutstr']} for data in data_list]
            for wt, label_info in zip(complex_wt_list_padded, label_list):
                data_list_padded.append({'complex': label_info['complex'], 'complex_wt': wt, 'labels': label_info['labels'], 'label_mask': label_info['label_mask'], 'mutstr': label_info['mutstr']})
        
        else: 
            data_list_padded = self.collate_complex(data_list)
    
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        if 'mut_seqs' in data_list[0]:
            prot_batch, mut_batch, prot_chains, prot_mask= self.pad_for_esm(data_list)
            batch['prot_mut'] = mut_batch
        else:
            prot_batch, prot_chains, prot_mask = self.pad_for_esm(data_list)
        batch['prot_seqs'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        return batch


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    dataset = PDBRedoChainDataset(args.split)

    for data in tqdm(dataset, desc='Iterating'):
        pass
    print(data)
    print(f'[INFO] {len(dataset.clusters)} clusters in the entire dataset.')
    print(f'[INFO] {len(dataset)} samples in the split.')
