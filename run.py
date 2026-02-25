import json
import os

import fire
from pathlib import Path
from argparse import Namespace

import numpy as np
import yaml
import wandb
import time
from easydict import EasyDict
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from typing import Dict

import models.encoders

import pandas as pd
import pytorch_lightning as pl
import logging
from data.protein import restypes
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pl_modules import MutationDataModule, MutationModelModule, TransferModelModule, PretuneModule
def parse_yaml(yaml_dir):
    with open(yaml_dir, 'r') as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
        # args = Namespace(**config_dict)
    return config_dict
def init_pytorch_settings():
    # Multiprocess Setting to speedup dataloader
    torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)

class LightningRunner(object):
    def __init__(self, model_config='./config/models/train/ESSM.yaml', data_config='./config/datasets/generative/RDE/SKEMPIv2.yaml',
                 run_config='./config/runs/train_basic.yaml'):
        super(LightningRunner, self).__init__()
        self.model_args = parse_yaml(model_config)
        self.dataset_args = parse_yaml(data_config)
        self.run_args = parse_yaml(run_config)
        init_pytorch_settings()

    def select_module(self, log_dir, k=None):
        if self.run_args.run_type in ['train', 'test', 'dms', 'light_train', 'light_test']:
            module = MutationModelModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args,
                                        run_args=self.run_args, fold=k)
            return module, MutationModelModule
        elif self.run_args.run_type in ['transfer', 'transfer_test', 'light_transfer', 'light_transfer_test']:
            module = TransferModelModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args,
                            run_args=self.run_args, fold=k)
            return module, MutationModelModule
        elif self.run_args.run_type == 'pretune':
            module = PretuneModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args,
                            run_args=self.run_args)
            return module, PretuneModule
        else:
            raise NotImplementedError


    def select_data_module(self, k=0):
        data_module = MutationDataModule(dataset_args=self.dataset_args, **self.dataset_args, col_group=f'fold_{k}')
        return data_module


    def pretrain(self):
        pass


    def train(self):
        print("Args:", self.run_args, self.dataset_args, self.model_args)
        output_dir, ckpt, gpus = (self.run_args.output_dir, self.run_args.ckpt,
                                             self.run_args.gpus)
        run_results = []
        # Setup datamodule
        for k in range(self.run_args.num_folds):
            print(f"Training fold {k} Started!")
            output_dir = Path(output_dir)
            log_dir = output_dir / f'log_fold_{k}'
            data_module = self.select_data_module(k)
            # data_module.setup()
            # Setup model module
            model, cls = self.select_module(log_dir, k)
            # Trainer setting
            # logger_csv = CSVLogger(str(log_dir))
            name = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
            if self.run_args.wandb:
                wandb.init(project='unippi', name=name)
                logger = WandbLogger()
            else:
                logger = CSVLogger(str(log_dir))
            # version_dir = Path(logger_csv.log_dir)
            pl.seed_everything(self.model_args.train.seed)
            print("Max epochs:", self.run_args.epochs)
            if self.run_args.ckpt is not None:
                ckpt = torch.load(self.run_args.ckpt)
                model.load_state_dict(ckpt['state_dict'])
            trainer = pl.Trainer(
                devices=gpus,
                # max_steps=self.run_args.iters,
                num_sanity_val_steps=0,
                max_epochs=self.run_args.epochs,
                logger=logger,
                # check_val_every_n_epoch=5,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=self.run_args.patience),
                    ModelCheckpoint(dirpath=(log_dir / 'checkpoint'), filename='{epoch}-{val_loss:.3f}',
                                    monitor="val_loss", mode="min", save_last=True, save_top_k=1),
                    # pl.callbacks.LambdaCallback(
                    #     on_validation_epoch_start=lambda trainer, pl_module: trainer.model.to(gpus[0])
                    # )
                    # ModelSummary(max_depth=2)
                    # TQDMProgressBar(refresh_rate=1)
                ],
                # gradient_clip_val=self.model_args.train.max_grad_norm if self.model_args.train.max_grad_norm is not None else None,
                # gradient_clip_algorithm='norm' if self.model_args.train.max_grad_norm is not None else None,
                strategy=DDPStrategy(find_unused_parameters=True),
                log_every_n_steps=3
            )
            # trainer.fit(model=model, datamodule=data_module, ckpt_path=self.run_args.ckpt)
            trainer.fit(model=model, datamodule=data_module)
            print(f"Training fold {k} Finished!")
            trainer.strategy.barrier()
            print("Best Validation Results:")
            _ = trainer.validate(model=model, ckpt_path="best", datamodule=data_module)
            res = model.res[0]
            run_results.append(res)
            if trainer.global_rank == 0:
                # self.save_model(cls, output_dir, trainer)
                torch.save(model.model.state_dict(), Path(output_dir) / 'best_model.pth')
                with open(Path(output_dir) / 'best_model_records.txt', mode='a') as f:
                    f.write(trainer.checkpoint_callback.best_model_path + '\n')
        result_dir = Path(output_dir) / name
        os.makedirs(result_dir, exist_ok=True)
        with open(result_dir / 'res.json', 'w') as f:
            json.dump(run_results, f)
        results_df = pd.DataFrame(run_results)
        print(results_df.describe())

                    
    def test(self):
        print("Args:", self.run_args, self.dataset_args, self.model_args)
        output_dir, ckpts, gpus = (self.run_args.output_dir, self.run_args.ckpts,
                                   self.run_args.gpus)
        name = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
        run_results = defaultdict(list)
        for k in range(self.run_args.num_folds):
            output_dir = Path(output_dir)
            log_dir = output_dir / f'log_fold_{k}'
            data_module = self.select_data_module(k)
            # data_module.setup()
            model, _ = self.select_module(log_dir, k)
            logger = CSVLogger(str(log_dir))
            trainer = pl.Trainer(
                num_sanity_val_steps=2,
                devices=gpus,
                max_epochs=0,
                logger=[
                    logger,
                ],
                callbacks=[
                    TQDMProgressBar(refresh_rate=1),
                ],
                strategy='ddp',
            )
            _ = trainer.test(model=model, ckpt_path=ckpts[k] if ckpts is not None else None, datamodule=data_module)
            res = model.res
            for i in range(len(res)):
                run_results[i].append(res[i])
        if trainer.global_rank == 0:
            result_dir = Path(output_dir) / name
            os.makedirs(result_dir, exist_ok=True)
            for i in run_results:
                with open(result_dir / 'res_task{}.json'.format(i), 'w') as f:
                    json.dump(run_results[i], f)
                results_df = pd.DataFrame(run_results[i])
                print(results_df.describe())


    def fold(self, input_dir, folding_args):
        pass

    def draw_heatmap(self, df_dms, period_list_all, output_dir):
        # matrix = np.zeros((end-start+1, 20))
        df_dms["index"] = df_dms['mutation'].str.extract('(\d+)').astype(int)
        for start, end in period_list_all:
            df_dms_period = df_dms[(df_dms['index'] >= start) & (df_dms['index'] <= end)]
            ddg_pred = df_dms_period['DDG_pred']
            muts = df_dms_period['mutation']
            ddg_full = []
            count = 0
            curr_pos = start
            for ddg, mut in zip(ddg_pred, muts):
                wt, pos, mt = mut[0], int(mut[2:-1]), mut[-1]
                idx = restypes.index(wt)
                if pos != curr_pos:
                    ddg_full += [0] * 20 * (pos - curr_pos)
                    curr_pos = pos
                if idx == count:
                    ddg_full.append(0)
                    ddg_full.append(ddg)
                else:
                    ddg_full.append(ddg)
                count += 1
                if count == 19:
                    if idx == 19:
                        ddg_full.append(0)
                    count = 0
                    curr_pos += 1
            if end != curr_pos:
                ddg_full += [0] * 20 * (end - curr_pos)
            ddg_full = np.array(ddg_full).reshape((-1, 20)).transpose()
            print(ddg_full.shape)
            indexs = [i for i in range(start, end + 1)]
            columns = restypes
            plt.figure(figsize=(max(ddg_full.shape[1] / 10, 2), ddg_full.shape[0] / 10))
            sns.heatmap(ddg_full, xticklabels=indexs, yticklabels=columns, cmap="RdBu_r")
            plt.xticks(rotation=45, fontsize=6)
            plt.yticks(fontsize=6)
            plt.savefig(output_dir / f'dms_{start}_{end}.jpg', dpi=400)


    # Deep Mutational Scaning
    def dms(self, input_dir, chain, partners, period=None, valid_chains=None,
            folding=False, folding_args=None, draw=False, positions=None):
        # print("Args:", self.run_args, self.dataset_args, self.model_args)
        output_dir, ckpts, gpus = (self.run_args.output_dir, self.run_args.ckpts, self.run_args.gpus)
        output_dir = Path(output_dir)
        log_dir = output_dir / 'log'
        logger = logging.getLogger('dms logger')
        os.makedirs(log_dir, exist_ok=True)
        logger.setLevel(logging.DEBUG)
        log_file = "{}/dms.log".format(log_dir)
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
        if folding:
            assert folding_args is not None
            assert '.fasta' in input_dir
            folding_args = parse_yaml(folding_args)
            logger.log("Folding {} with {}!".format(input_dir.split('/')[-1], folding_args.model))
            complex_dir = self.fold(input_dir, folding_args)
        else:
            assert '.pdb' in input_dir or '.cif' in input_dir
            complex_dir = input_dir
        data_module = MutationDataModule(dataset_args=self.dataset_args, **self.dataset_args)
        if positions is None:
            df_dms, period_list_all = data_module.dms_setup(complex_dir, chain, period, valid_chains, logger, partners, output_dir)
        else:
            df_dms = data_module.multi_mut_setup(complex_dir, chain, positions, valid_chains, logger, partners, output_dir)
            period_list_all = None
        # model = MutationModelModule(output_dir=output_dir, model_args=self.model_args, data_args=self.dataset_args,
        #                            run_args=self.run_args)
        models = []
        if ckpts is not None:
            for ckpt in ckpts:
                model = MutationModelModule.load_from_checkpoint(ckpt, map_location=torch.device(self.run_args.gpus[0]))
                # model.eval()
                models.append(model)
        else:
            print("Directly loading without checkpoints...")
            model, _ = self.select_module(log_dir, 0)
            # model.eval()
            model.model = model.model.to(self.run_args.gpus[0])
            models.append(model)

        dataloader = data_module.test_dataloader()
        
        output_list = []
        for batch in tqdm(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.run_args.gpus[0])
                elif isinstance(batch[key], Dict):
                    for key2 in batch[key]:
                        if isinstance(batch[key][key2], torch.Tensor):
                            batch[key][key2] = batch[key][key2].to(self.run_args.gpus[0])
            pred_ddgs = []
            for model in models: 
                with torch.no_grad():
                    pred_dict = model.model.inference(batch)
                pred_ddg = pred_dict['y_pred'].squeeze()
                pred_ddgs.append(pred_ddg)
            stacked_tensor = torch.stack(pred_ddgs)
            ens_pred_ddg = torch.mean(stacked_tensor, dim=0)
            output_list.append(ens_pred_ddg)
            # print(pred_dict, y, mask)
        if len(output_list[-1].shape) == 0:
            output_list[-1] = output_list[-1].reshape([1])
        dms_results = torch.cat(output_list)
        df_dms['DDG_pred'] = dms_results.cpu().numpy().tolist()
        pdb_id = complex_dir.split('/')[-1]
        if positions is None:
            df_dms.to_csv(output_dir/f"dms_{pdb_id}_{chain}_{period}.csv", index=False)
        else:
            with open(positions, 'r') as f:
                lines = f.readlines()
                num = len(lines[0].strip().split(','))
            df_dms.to_csv(output_dir/f"{num}_points_{pdb_id}_{chain}_{period}.csv", index=False)
        # df_dms = pd.read_csv(output_dir / 'dms_5h37.pdb_A_20,30;40,45.csv')

        if draw:
            logger.info('Drawing heatmap for this scan...')
            self.draw_heatmap(df_dms, period_list_all, output_dir)
            logger.info("Done, the heatmap is saved under the output directory.")
        

if __name__ == '__main__':
    fire.Fire(LightningRunner)