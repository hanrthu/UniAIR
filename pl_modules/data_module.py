import pandas as pd
import pytorch_lightning as pl
import diskcache
from data import PaddingCollate, inf_iterator, DataRegister
from torch.utils.data import DataLoader
from data.protein import ProteinInput, restypes, lexico_restypes
from collections import defaultdict
import itertools
import torch
import os

def get_dataset(data_args:dict=None):
    register = DataRegister()
    dataset_cls = register[data_args.dataset_type]
    return dataset_cls

class MutationDataModule(pl.LightningDataModule):
    def __init__(self, df_path='', max_length=256, n_neighbors=32, col_group='fold_0',
                 batch_size=32, num_workers=0, pin_memory=True, shuffle=True, cache_dir=None, dataset_args=None, **kwargs):
        """
        :param df_path:
        :param max_length:
        :param n_neighbors:
        :param cols_label:
        :param col_group:
        :param batch_size:
        :param num_workers:
        :param pin_memory:
        :param shuffle:
        :param cache_dir:
        :param kwargs:
        """
        super().__init__()
        self.df_path = df_path

        self.max_length = max_length
        self.n_neighbors = n_neighbors
        self.col_group = col_group
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.cache_dir = cache_dir
        self.dataset_args = dataset_args
        print('Dataset Args:', dataset_args)

    def prepare_data(self):
        pass

    def setup_pretrain(self, stage=None):
        if self.cache_dir is None:
            cache = None
        else:
            print("Using diskcache at {}.".format(self.cache_dir))
            cache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')

        df = pd.read_csv(self.df_path)
        dataset_cls = get_dataset(self.dataset_args)
        self.pretrain_dataset = dataset_cls(df, **self.dataset_args, diskcache=cache)

    def setup(self, stage=None,pretrain = None):
        if self.cache_dir is None:
            cache = None
        else:
            print("Using diskcache at {}.".format(self.cache_dir))
            cache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')
        df = pd.read_csv(self.df_path)
        df_train = df[df[self.col_group].isin(['train'])]
        df_val = df[df[self.col_group].isin(['val'])]
        df_test = df[df[self.col_group].isin(['test'])]
        dataset_cls = get_dataset(self.dataset_args)
        self.train_dataset = dataset_cls(df_train, **self.dataset_args, diskcache=cache)
        self.val_dataset = dataset_cls(df_val, **self.dataset_args, diskcache=cache)

        if len(df_test) > 0 :
            self.test_dataset = dataset_cls(df_test, **self.dataset_args, diskcache=cache)
        else:
            val_length = len(df_val)
            print(f"Using Validation Fold {self.col_group} to test the model!, total length: {val_length}")
            self.test_dataset = dataset_cls(df_val, **self.dataset_args, diskcache=cache)

    def dms_setup(self, complex_dir, chain_id, period, valid_chains, logger, partners, output_dir):
        dms_headers = ['Protein', 'pdb_id', 'Partners(A_B)', 'mutation', 'path_wt', 'path_mut', 'valid_chains', 'DDG']
        dms_data = []
        protein = ProteinInput.from_path(complex_dir, with_angles=False, return_dict=True)
        pdb_id = complex_dir.split('/')[-1]
        logger.info(f"Loaded protein from {complex_dir}")
        chain = protein[chain_id]
        seq_map = {}
        for i, (chain_id, resseq) in enumerate(zip(chain.chain_id, chain.resseq)):
            seq_map[(chain_id, int(resseq))] = i
        # print(seq_map)
        period_list_all = []

        if period is None:
            dms_period_list = [None]
        else:
            if ";" in period:
                dms_period_list = period.split(";")
            else:
                if isinstance(period, tuple):
                    dms_period_list = [','.join(map(str, period))]
        # print(dms_period_list)
        print("DMS_period_list:", dms_period_list)
        for dms_period in dms_period_list:
            if dms_period is None:
                start, end = int(chain.resseq[0]), int(chain.resseq[-1])
            else:
                start, end = dms_period.split(",")
                start, end = int(start), int(end)
            if start is None:
                start = int(chain.resseq[0])
            if end is None:
                end = int(chain.resseq[-1])
            period_list_all.append([start, end])
            for i in range(start, end + 1):
                if (chain_id, i) in seq_map:
                    res_idx = seq_map[(chain_id, i)]
                else:
                    continue
                aa_wt = chain.seq[res_idx]
                for j in lexico_restypes[:-1]:
                    if j == aa_wt:
                        continue
                    else:
                        aa_mt = j
                        mut_str = f'{aa_wt}{chain_id}{i}{aa_mt}'
                        suffix = f'_{mut_str}.pdb'
                        path_mut = complex_dir.replace('PDBs', 'PDBs_mt').replace('.pdb', suffix)
                        data_item = [pdb_id[:-4], pdb_id, partners, mut_str, complex_dir, path_mut, valid_chains, 0]
                        dms_data.append(data_item)

        df_dms = pd.DataFrame(dms_data, columns=dms_headers)
        df_dms.to_csv('./tmp_dms.csv')
        logger.info(f"Number of scanning for DMS: {len(df_dms)}")
        dataset_cls = get_dataset(self.dataset_args)
        self.test_dataset = dataset_cls(df_dms, **self.dataset_args)
        return df_dms, period_list_all
    
    
    def multi_mut_setup(self, complex_dir, chain_id, positions, valid_chains, logger, partners, output_dir):
        dms_headers = ['Protein', 'pdb_id', 'Partners(A_B)', 'mutation', 'path_wt', 'path_mut', 'valid_chains', 'DDG']
        dms_data = []
        protein = ProteinInput.from_path(complex_dir, with_angles=False, return_dict=True)
        pdb_id = complex_dir.split('/')[-1]
        logger.info(f"Loaded protein from {complex_dir}")
        chain = protein[chain_id]
        seq_map = {}
        for i, (chain_id, resseq) in enumerate(zip(chain.chain_id, chain.resseq)):
            seq_map[(chain_id, int(resseq))] = i
        
        if positions is None:
            raise ValueError("Specification of positions is necessary")
        else:
            if "," in positions:
                dms_position_list = positions.split(";")
            elif '.txt' in positions:
                with open(positions, 'r') as f:
                    lines = f.readlines()
                    dms_position_list = [line.strip() for line in lines]
            else:
                if isinstance(positions, tuple):
                    dms_position_list = list(positions)
        # print(dms_period_list)
        print("DMS_period_list:", dms_position_list, len(dms_position_list))
        mut_str_dict = {}

        for n, dms_positions in enumerate(dms_position_list):
            mut_str_dict[n] = defaultdict(list)
            for i, dms_position in enumerate(dms_positions.split(',')):
                # print('Chain:', chain_id, dms_position)
                dms_position = int(dms_position)
                if (chain_id, dms_position) in seq_map:
                    res_idx = seq_map[(chain_id, dms_position)]
                else:
                    continue
                aa_wt = chain.seq[res_idx]
                for j in lexico_restypes[:-1]:
                    if j == aa_wt:
                        continue
                    else:
                        aa_mt = j
                        mut_str = f'{aa_wt}{chain_id}{dms_position}{aa_mt}'
                        mut_str_dict[n][i].append(mut_str)
                        # suffix = f'_{mut_str}.pdb'
                        # path_mut = complex_dir.replace('PDBs', 'PDBs_mt').replace('.pdb', suffix)
                        # data_item = [pdb_id[:-4], pdb_id, partners, mut_str, complex_dir, path_mut, valid_chains, 0]
                        # dms_data.append(data_item)
        mutstrs = []
        for key in mut_str_dict:
            sub_dict = mut_str_dict[key]
            lists = list(sub_dict.values())
            print(mut_str_dict)
            sub_mutstr = [",".join(item) for item in itertools.product(*lists)]
            mutstrs.extend(sub_mutstr)
        for mutstr in mutstrs:
            suffix = f'_{mutstr}.pdb'
            path_mut = complex_dir.replace('PDBs', 'PDBs_mt').replace('.pdb', suffix)
            data_item = [pdb_id[:-4], pdb_id, partners, mutstr, complex_dir, path_mut, valid_chains, 0]
            dms_data.append(data_item)
        df_dms = pd.DataFrame(dms_data, columns=dms_headers)
        df_dms.to_csv('./tmp_multipoints.csv')
        logger.info(f"Number of scanning for DMS: {len(df_dms)}")
        dataset_cls = get_dataset(self.dataset_args)
        self.test_dataset = dataset_cls(df_dms, **self.dataset_args)
        return df_dms


    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=PaddingCollate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=PaddingCollate(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=PaddingCollate(),
        )

    def pretrain_dataloader(self):
        return DataLoader(
            self.pretrain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=PaddingCollate(),
        )

class LightMutationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, data_pred_dir=None, max_length=256, n_neighbors=32, fold='fold_0',
                 batch_size=32, num_workers=0, pin_memory=True, shuffle=True, cache_dir=None, dataset_args=None, **kwargs):
        """
        :param df_path:
        :param max_length:
        :param n_neighbors:
        :param cols_label:
        :param col_group:
        :param batch_size:
        :param num_workers:
        :param pin_memory:
        :param shuffle:
        :param cache_dir:
        :param kwargs:
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_pred_dir = data_pred_dir

        self.max_length = max_length
        self.n_neighbors = n_neighbors
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.cache_dir = cache_dir
        self.dataset_args = dataset_args
        print('Dataset Args:', dataset_args)

    def setup(self, stage=None, pretrain = None):
        if self.cache_dir is None:
            cache = None
        else:
            print("Using diskcache at {}.".format(self.cache_dir))
            cache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')

        # data = torch.load(self.data_dir, map_location='cpu')
        train_dir = os.path.join(self.data_dir, 'train', self.fold)
        val_dir = os.path.join(self.data_dir, 'val', self.fold)
        dataset_cls = get_dataset(self.dataset_args)
        self.train_dataset = dataset_cls(train_dir, **self.dataset_args, diskcache=cache)
        self.val_dataset = dataset_cls(val_dir, **self.dataset_args, diskcache=cache)
        self.test_dataset = dataset_cls(val_dir, **self.dataset_args, diskcache=cache)
        print(f"Using Validation Fold {self.fold} to test the model!")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=LightCollate(),
            # collate_fn=PaddingCollate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=LightCollate(),
            # collate_fn=PaddingCollate(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=LightCollate(),
            # collate_fn=PaddingCollate(),
        )    
    
# Pretraining DataModule, no cross validation
class RotamerPretrainDataModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()
        self.cfg = data_args
        self.batch_size = data_args.batch_size
        self.num_workers = data_args.num_workers
        self.pin_memory = data_args.pin_memory

    def setup(self, stage=None):
        self.train_dataset = PDBRedoChainDataset('train', self.cfg.pdbredo_dir, self.cfg.clusters_path, self.cfg.splits_path, self.cfg.processed_dir, transform = self.cfg.transform)
        self.val_dataset = PDBRedoChainDataset('val', self.cfg.pdbredo_dir, self.cfg.clusters_path, self.cfg.splits_path, self.cfg.processed_dir, transform = self.cfg.transform)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=PaddingCollate()
        )
        return inf_iterator(train_loader)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=PaddingCollate()
        )
