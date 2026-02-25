import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models import ModelRegister
from utils.metrics import ScalarMetricAccumulator, per_complex_corr, per_complex_acc, \
                            cal_accuracy, cal_precision, cal_recall, cal_pearson, cal_spearman, \
                            cal_rmse, cal_mae, cal_weighted_loss, sum_weighted_losses, cal_auc
def get_model(model_args:dict=None, output_dim=1, fold=None):
    register = ModelRegister()
    # print("Items:", register)
    if fold != None:
        model_args_ori = {'output_dim': output_dim, 'fold': fold}
    else:
        model_args_ori = {'output_dim': output_dim}
    model_args_ori.update(model_args)
    model_cls = register[model_args['model_type']]
    model = model_cls(**model_args_ori)
    if 'state_dict' in model_args and model_args['state_dict'] is not None:
        model.load_state_dict(torch.load(model_args['state_dict']))
        print(f"Successfully loaded state dict for {model_args['model_type']}")

    return model

class MutationModelModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None, fold=None):
        super().__init__()
        self.save_hyperparameters()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = len(data_args.cols_label)
        model_args.model.output_dim = self.num_classes
        self.model = get_model(model_args=model_args.model, output_dim=len(data_args.loss_weights), fold=fold)
        # self.reg_weight = model_args.get('reg_weight', np.ones(self.num_classes) * 0.05)
        self.class_dir = model_args.get('class_dir', np.ones(self.num_classes))
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        self.valid_it = 0
        self.batch_size = data_args.batch_size
        self.train_loss = None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        if self.optimizers_cfg.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.optimizers_cfg.lr, 
                                         betas=(self.optimizers_cfg.beta1, self.optimizers_cfg.beta2, ))
        elif self.optimizers_cfg.type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizers_cfg.lr)
        elif self.optimizers_cfg.type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizers_cfg.lr)
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizers_cfg.type)

        if self.scheduler_cfg.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.scheduler_cfg.factor, 
                                                                   patience=self.scheduler_cfg.patience, 
                                                                   min_lr=self.scheduler_cfg.min_lr)
        elif self.scheduler_cfg.type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=self.scheduler_cfg.milestones, 
                                                             gamma=self.scheduler_cfg.gamma)
        elif self.scheduler_cfg.type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                               gamma=self.scheduler_cfg.gamma)
        else:
            raise NotImplementedError('Scheduler not supported: %s' % self.scheduler_cfg.type)

        if self.model_args.resume is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch.load(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print(
                'Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
            
        if self.scheduler_cfg.type == 'plateau':
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }
            }
        else:
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                }
            }
        return optim_dict

    def on_train_start(self):
        log_hyperparams = {'model_args':self.model_args, 'data_args': self.data_args, 'run_args': self.run_args}
        self.logger.log_hyperparams(log_hyperparams)

    def _get_reg_loss(self, y_pred, y, mask):
        # 'reg' means regulation, mapping the prediction of ddG to other tasks with a monotonic layer.
        y_pred = self.regular_layer(y_pred)
        pred_dict = {
            'y_pred': y_pred[..., 1:]
        }
        loss_reg = cal_weighted_loss(pred_dict, y[..., 1:], mask[..., 1:], self.data_args.loss_types[1:], self.data_args.loss_weights[1:])
        return loss_reg

    def on_before_optimizer_step(self, optimizer) -> None:
        pass
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        #         print("Found Unused Parameters")

    def training_step(self, batch, batch_idx):
        y = batch['labels']
        mask = batch['label_mask']
        pred_dict = self.model.inference(batch)
        # if self.add_pretrain:
        #     pred_dict = self.model(batch,self.pretrain_model)
        # else:
            # pred_dict = self.model(batch)
        loss = cal_weighted_loss(pred_dict, y, mask, self.data_args.loss_types, self.data_args.loss_weights)
        if 'aux_loss' in pred_dict:
            loss += pred_dict['aux_loss'] * pred_dict['aux_weight']

        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        pred_dict = self.model.inference(batch)
        y = batch['labels']
        mask = batch['label_mask']
        val_loss = cal_weighted_loss(pred_dict, y, mask, self.data_args.loss_types, self.data_args.loss_weights)

        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=len(batch['labels']), mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for complex, mutstr, y_true, y_mask, y_pred in zip(batch['complex'], batch['mutstr'], batch['labels'], batch['label_mask'],
                                                       pred_dict['y_pred']):
            result = {}
            if len(self.data_args.loss_types) == 1:
                result['y_true_{}'.format(0)] = y_true.item()
                result['y_pred_{}'.format(0)] = y_pred.item()
                result['y_mask_{}'.format(0)] = y_mask.item()
            else:
                for i, l_type in enumerate(self.data_args.loss_types):
                    result['y_true_{}'.format(i)] = y_true[i].item()
                    result['y_pred_{}'.format(i)] = y_pred[i].item()
                    result['y_mask_{}'.format(i)] = y_mask[i].item()
            result['complex'] = complex
            result['mutstr'] = mutstr
            result['num_muts'] = len(mutstr.split(','))
            self.results.append(result)
        return val_loss
    
    def on_validation_epoch_end(self):
        results = pd.DataFrame(self.results)
        res_list = []
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
        for i, l_type in enumerate(self.data_args.loss_types):
            y_pred_i = np.array(results[f'y_pred_{i}'])
            y_true_i = np.array(results[f'y_true_{i}'])
            y_mask_i = np.array(results[f'y_mask_{i}'])
            if y_mask_i.sum() == 0:
                continue
            else:
                y_pred_i = y_pred_i[y_mask_i == 1]
                y_true_i = y_true_i[y_mask_i == 1]
            pearson_all = cal_pearson(y_pred_i, y_true_i)
            spearman_all = cal_spearman(y_pred_i, y_true_i)
            rmse_all = cal_rmse(y_pred_i, y_true_i)
            mae_all = cal_mae(y_pred_i, y_true_i)
            pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred_{}'.format(i), true_attr='y_true_{}'.format(i))
            print(f'[All_Task_{i}] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
            print(f'[PC_Task_{i}]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
            
            self.log(f'val/all_pearson_{i}', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_spearman_{i}', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_rmse_{i}', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_mae_{i}', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/pc_pearson_{i}', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/pc_spearman_{i}', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/pc_rmse_{i}', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/pc_mae_{i}', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all, 'pc pearson': pearson_pc, 'pc spearman': spearman_pc}
            res_list.append(res)
        self.res = res_list
        val_loss = self.scalar_accum.get_average('val_loss')
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()
        
    def test_step(self, batch, batch_idx):
        # print('Batch:', batch)
        pred_dict = self.model.inference(batch)
        y = batch['labels']
        mask = batch['label_mask']
        test_loss = cal_weighted_loss(pred_dict, y, mask, self.data_args.loss_types, self.data_args.loss_weights)
        # print("Mask:", mask, mask.shape)
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=len(batch['labels']), mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for complex, mutstr, y_true, y_mask, y_pred in zip(batch['complex'], batch['mutstr'], batch['labels'],
                                                           batch['label_mask'],
                                                           pred_dict['y_pred']):
            result = {}
            if len(self.data_args.loss_types) == 1:
                result['y_true_{}'.format(0)] = y_true.item()
                result['y_pred_{}'.format(0)] = y_pred.item()
                result['y_mask_{}'.format(0)] = y_mask.item()
            else:
                for i, l_type in enumerate(self.data_args.loss_types):
                    result['y_true_{}'.format(i)] = y_true[i].item()
                    result['y_pred_{}'.format(i)] = y_pred[i].item()
                    result['y_mask_{}'.format(i)] = y_mask[i].item()
            result['complex'] = complex
            result['mutstr'] = mutstr
            result['num_muts'] = len(mutstr.split(','))
            self.results.append(result)
        return test_loss

    def on_test_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_test.csv'), index=False)
        res_list = []
        for i, l_type in enumerate(self.data_args.loss_types):
            y_pred_i = np.array(results[f'y_pred_{i}'])
            y_true_i = np.array(results[f'y_true_{i}'])
            y_mask_i = np.array(results[f'y_mask_{i}'])
            if y_mask_i.sum() == 0:
                continue
            else:
                y_pred_i = y_pred_i[y_mask_i == 1]
                y_true_i = y_true_i[y_mask_i == 1]
            pearson_all = cal_pearson(y_pred_i, y_true_i)
            spearman_all = cal_spearman(y_pred_i, y_true_i)
            rmse_all = cal_rmse(y_pred_i, y_true_i)
            mae_all = cal_mae(y_pred_i, y_true_i)
            try:
                pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred_{}'.format(i), true_attr='y_true_{}'.format(i))
                print(f'[All_Task_{i}] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
                print(f'[PC_Task_{i}]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
            
                self.log(f'test/all_pearson_{i}', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_spearman_{i}', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_rmse_{i}', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_mae_{i}', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_pearson_{i}', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_spearman_{i}', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_rmse_{i}', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_mae_{i}', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all, 'pc pearson': pearson_pc, 'pc spearman': spearman_pc}
            except:
                print(f'[All_Task_{i}] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
                self.log(f'test/all_pearson_{i}', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_spearman_{i}', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_rmse_{i}', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_mae_{i}', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all}
            res_list.append(res)
        self.res = res_list
        test_loss = self.scalar_accum.get_average('loss')
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        return test_loss    


class TransferModelModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None, fold=None):
        super().__init__()
        self.save_hyperparameters()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = len(data_args.cols_label)
        model_args.model.output_dim = self.num_classes
        self.model = get_model(model_args=model_args.model, output_dim=len(data_args.loss_weights), fold=fold)
        self.class_dir = model_args.get('class_dir', np.ones(self.num_classes))
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        self.valid_it = 0
        self.batch_size = data_args.batch_size
        self.train_loss = None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        if self.optimizers_cfg.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.optimizers_cfg.lr, 
                                         betas=(self.optimizers_cfg.beta1, self.optimizers_cfg.beta2, ))
        elif self.optimizers_cfg.type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizers_cfg.lr)
        elif self.optimizers_cfg.type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizers_cfg.lr)
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizers_cfg.type)

        if self.scheduler_cfg.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.scheduler_cfg.factor, 
                                                                   patience=self.scheduler_cfg.patience, 
                                                                   min_lr=self.scheduler_cfg.min_lr)
        elif self.scheduler_cfg.type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=self.scheduler_cfg.milestones, 
                                                             gamma=self.scheduler_cfg.gamma)
        elif self.scheduler_cfg.type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                               gamma=self.scheduler_cfg.gamma)
        else:
            raise NotImplementedError('Scheduler not supported: %s' % self.scheduler_cfg.type)

        if self.model_args.resume is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch.load(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print(
                'Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
            
        if self.scheduler_cfg.type == 'plateau':
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }
            }
        else:
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                }
            }
        return optim_dict

    def on_train_start(self):
        log_hyperparams = {'model_args':self.model_args, 'data_args': self.data_args, 'run_args': self.run_args}
        self.logger.log_hyperparams(log_hyperparams)

    def on_before_optimizer_step(self, optimizer) -> None:
        pass
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        #         print("Found Unused Parameters")

    def training_step(self, batch, batch_idx):
        
        loss_dict = self.model(batch)
        loss = loss_dict['H'] + self.model_args.model.alpha * loss_dict['X']
        if 'DDG' in loss_dict:
            loss += loss_dict['DDG']

        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.results = []
        self.ddg = False
        self.scalar_accum = ScalarMetricAccumulator()

    def validation_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        val_loss = loss_dict['H'] + self.model_args.model.alpha * loss_dict['X']
        if 'DDG' in loss_dict:
            self.ddg = True
            val_loss += loss_dict['DDG']
            pred_dict = self.model.inference(batch)
            y = batch['labels']
            mask = batch['label_mask']
            val_loss = cal_weighted_loss(pred_dict, y, mask, self.data_args.loss_types, self.data_args.loss_weights)

            # if self.num_classes > 1:
            #     loss_reg = self._get_reg_loss(pred_dict['y_pred'], y, mask)
            #     val_loss = val_loss + loss_reg

            self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=batch['size'], mode='mean')
            self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            for complex, mutstr, y_true, y_mask, y_pred in zip(batch['complex'], batch['mutstr'], batch['labels'], batch['label_mask'],
                                                        pred_dict['y_pred']):
                result = {}
                result['y_true_{}'.format(0)] = y_true.item()
                result['y_pred_{}'.format(0)] = y_pred.item()
                result['y_mask_{}'.format(0)] = y_mask.item()
                result['complex'] = complex
                result['mutstr'] = mutstr
                result['num_muts'] = len(mutstr.split(','))
                self.results.append(result)
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=batch['size'], mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss
    
    def on_validation_epoch_end(self):
        val_loss = self.scalar_accum.get_average('val_loss')
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        # print("Val Loss:", val_loss)
        self.valid_it += 1
        res = {"val_loss": val_loss}
        if self.ddg:
            results = pd.DataFrame(self.results)
            res_list = []
            if self.output_dir is not None:
                results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
                y_pred_i = np.array(results[f'y_pred_0'])
                y_true_i = np.array(results[f'y_true_0'])
                y_mask_i = np.array(results[f'y_mask_0'])
                y_pred_i = y_pred_i[y_mask_i == 1]
                y_true_i = y_true_i[y_mask_i == 1]
                pearson_all = cal_pearson(y_pred_i, y_true_i)
                spearman_all = cal_spearman(y_pred_i, y_true_i)
                rmse_all = cal_rmse(y_pred_i, y_true_i)
                mae_all = cal_mae(y_pred_i, y_true_i)
                pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred_{}'.format(0), true_attr='y_true_{}'.format(0))
                print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
                print(f'[PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
                
                self.log(f'val/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all, 'pc pearson': pearson_pc, 'pc spearman': spearman_pc}
                res_list.append(res)
                    
        self.res = [res]
        return val_loss

    def on_test_epoch_start(self):
        self.results = []
        self.ddg = False
        self.scalar_accum = ScalarMetricAccumulator()

    def test_step(self, batch, batch_idx):
        self.ddg = True
        pred_dict = self.model.inference(batch)
        y = batch['labels']
        mask = batch['label_mask']
        test_loss = cal_weighted_loss(pred_dict, y, mask, self.data_args.loss_types, self.data_args.loss_weights)

        # if self.num_classes > 1:
        #     loss_reg = self._get_reg_loss(pred_dict['y_pred'], y, mask)
        #     val_loss = val_loss + loss_reg

        self.scalar_accum.add(name='test_loss', value=test_loss, batchsize=batch['size'], mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for complex, mutstr, y_true, y_mask, y_pred in zip(batch['complex'], batch['mutstr'], batch['labels'], batch['label_mask'],
                                                    pred_dict['y_pred']):
            result = {}
            result['y_true_{}'.format(0)] = y_true.item()
            result['y_pred_{}'.format(0)] = y_pred.item()
            result['y_mask_{}'.format(0)] = y_mask.item()
            result['complex'] = complex
            result['mutstr'] = mutstr
            result['num_muts'] = len(mutstr.split(','))
            self.results.append(result)
        self.scalar_accum.add(name='test_loss', value=test_loss, batchsize=batch['size'], mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return test_loss
    
    def on_test_epoch_end(self):
        test_loss = self.scalar_accum.get_average('test_loss')
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        # print("Val Loss:", val_loss)
        # self.test_it += 1
        res = {"test_loss": test_loss}
        if self.ddg:
            results = pd.DataFrame(self.results)
            res_list = []
            if self.output_dir is not None:
                results.to_csv(os.path.join(self.output_dir, f'results_test.csv'), index=False)
                y_pred_i = np.array(results[f'y_pred_0'])
                y_true_i = np.array(results[f'y_true_0'])
                y_mask_i = np.array(results[f'y_mask_0'])
                y_pred_i = y_pred_i[y_mask_i == 1]
                y_true_i = y_true_i[y_mask_i == 1]
                pearson_all = cal_pearson(y_pred_i, y_true_i)
                spearman_all = cal_spearman(y_pred_i, y_true_i)
                rmse_all = cal_rmse(y_pred_i, y_true_i)
                mae_all = cal_mae(y_pred_i, y_true_i)
                pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred_{}'.format(0), true_attr='y_true_{}'.format(0))
                print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
                print(f'[PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
                
                self.log(f'test/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all, 'pc pearson': pearson_pc, 'pc spearman': spearman_pc}
                res_list.append(res)
                    
        self.res = [res]
        return test_loss


class PretuneModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None):
        super().__init__()
        self.save_hyperparameters()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.l_type = data_args.loss_types
        self.model = get_model(model_args=model_args.model)
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        self.valid_it = 0
        self.temperature = model_args.train.temperature
        self.batch_size = data_args.batch_size

        self.train_loss = None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        if self.optimizers_cfg.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.optimizers_cfg.lr, 
                                         betas=(self.optimizers_cfg.beta1, self.optimizers_cfg.beta2, ))
        elif self.optimizers_cfg.type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizers_cfg.lr)
        elif self.optimizers_cfg.type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizers_cfg.lr)
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizers_cfg.type)

        if self.scheduler_cfg.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.scheduler_cfg.factor, 
                                                                   patience=self.scheduler_cfg.patience, 
                                                                   min_lr=self.scheduler_cfg.min_lr)
        elif self.scheduler_cfg.type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=self.scheduler_cfg.milestones, 
                                                             gamma=self.scheduler_cfg.gamma)
        elif self.scheduler_cfg.type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                               gamma=self.scheduler_cfg.gamma)
        else:
            raise NotImplementedError('Scheduler not supported: %s' % self.scheduler_cfg.type)

        if self.model_args.resume is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch.load(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print(
                'Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
            
        if self.scheduler_cfg.type == 'plateau':
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }
            }
        else:
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                }
            }
        return optim_dict

    def on_train_start(self):
        log_hyperparams = {'model_args':self.model_args, 'data_args': self.data_args, 'run_args': self.run_args}
        self.logger.log_hyperparams(log_hyperparams)

    def on_before_optimizer_step(self, optimizer) -> None:
        pass
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        #         print("Found Unused Parameters")
    
    def continuous_to_discrete_tensor(self, values):
        discrete_values = torch.zeros_like(values, dtype=torch.long, device=values.device)
        
        mask_0_8 = values < 8
        discrete_values[mask_0_8] = (values[mask_0_8] * 2 + 0.5).long()
        
        mask_8_32 = (values >= 8) & (values < 32)
        discrete_values[mask_8_32] = (16 + (values[mask_8_32] - 8)).long()
        
        mask_ge_32 = (values >= 32)
        discrete_values[mask_ge_32] = 39
        discrete_values = torch.clamp(discrete_values, 0, 39).long()
        return discrete_values
    
    def cal_loss(self, pred_clip, y_clip, pred_dist, y_dist, identifier):
        # y_dist = torch.clamp(y_dist.long(), 0, 31).long()
        y_dist = self.continuous_to_discrete_tensor(y_dist)
        total_y_dists = []
        total_pred_dists = []
        total_y_dists_inv = []
        total_pred_dists_inv = []
        for i in range(identifier.shape[0]):
            item_identifier = identifier[i].squeeze()
            y_tmp = y_dist[i, item_identifier==0]
            y_interface_dist = y_tmp[:, item_identifier==1].flatten()
            total_y_dists.append(y_interface_dist)
            
            y_pred_tmp = pred_dist[i, item_identifier==0]
            y_interface_pred = y_pred_tmp[:, item_identifier==1].reshape([-1, pred_dist.shape[-1]])
            total_pred_dists.append(y_interface_pred)
            
            y_tmp_inv = y_dist[i, item_identifier==1]
            y_interface_dist_inv = y_tmp_inv[:, item_identifier==0].flatten()
            total_y_dists_inv.append(y_interface_dist_inv)
            
            y_pred_tmp_inv = pred_dist[i, item_identifier==1]
            y_interface_pred_inv = y_pred_tmp_inv[:, item_identifier==0].reshape([-1, pred_dist.shape[-1]])
            total_pred_dists_inv.append(y_interface_pred_inv)
            
        total_pred_dists = torch.cat(total_pred_dists, dim=0)
        total_pred_dists_inv = torch.cat(total_pred_dists_inv, dim=0)
        total_y_dists = torch.cat(total_y_dists)
        total_y_dists_inv = torch.cat(total_y_dists_inv)
        # y_dist_one_hot = F.one_hot(indices, num_classes=32).float()
        loss_clip_forward = F.cross_entropy(pred_clip / self.temperature, y_clip.long())
        loss_clip_inverse = F.cross_entropy(pred_clip.transpose(0, 1) / self.temperature, y_clip.long())
        loss_clip = 0.5 * (loss_clip_forward + loss_clip_inverse)
        # print("Loss CLIP:", loss_clip)
        # loss_dist_forward = F.cross_entropy(y_interface_pred.permute(0, 3, 1, 2) / self.temperature , y_interface_dist.long())
        # loss_dist_inv = F.cross_entropy(y_interface_pred_inv.permute(0, 3, 1, 2) / self.temperature , y_interface_dist_inv.long())
        loss_dist_forward = F.cross_entropy(total_pred_dists / self.temperature , total_y_dists.long())
        loss_dist_inverse = F.cross_entropy(total_pred_dists_inv / self.temperature, total_y_dists_inv.long())
        
        loss_dist = 0.5 * (loss_dist_forward + loss_dist_inverse)
        # print("Loss Dist:", loss_dist)
        return loss_clip, loss_dist
    
    def training_step(self, batch, batch_idx):
        # y_dist = batch['atom_min_dist']
        atoms = batch['complex_wt']['pos_atoms']
        mask_atoms = batch['complex_wt']['mask_atoms']
        distance_map = torch.linalg.norm(atoms[:, :, None, :, None, :]- atoms[:, None, :, None, :, :], dim=-1, ord=2).reshape(batch['size'], atoms.shape[1], atoms.shape[1], -1)
        mask = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(batch['size'], atoms.shape[1], atoms.shape[1], -1)
        distance_map[~mask] = torch.inf
        y_dist = torch.min(distance_map, dim=-1)[0]
        # print(distance_map.shape, y_dist.shape, atoms.shape)
        y_dg = batch['labels'].squeeze(1)
        dg_mask = y_dg != 0
        # res_identifier = batch['identifier']
        res_identifier = batch['complex_wt']['group_id'] - 1
        res_identifier[res_identifier < 0] = 0
        # y_clip = batch['clip_label']
        y_clip = torch.arange(batch['size'], dtype=torch.long, device=y_dist.device)
        pred_dg, pred_dist, pred_clip = self.model(batch, stage='pretune', need_mask=True)
        if dg_mask.sum() == 0:
            loss_dg = F.mse_loss(pred_dg, y_dg) * 0
        else:
            loss_dg = F.mse_loss(pred_dg, y_dg, reduction = 'none')
            loss_dg = (loss_dg * dg_mask).sum() / (dg_mask.sum() + 1e-5)
        loss_clip, loss_dist = self.cal_loss(pred_clip, y_clip, pred_dist, y_dist, res_identifier)
        loss = loss_clip + loss_dist + 10 * loss_dg
        skip_step = torch.tensor(int(torch.isnan(loss).any()), device=self.device)
        # 同步所有卡的 skip_step，最大值决定是否跳过
        skip_step = self.all_gather(skip_step).max()  # 使用 all_gather 确保所有卡同步

        if skip_step.item() == 1:
            print(f"Found NaN in loss at batch {batch_idx}, skipping this batch.")
            loss = 0 * loss
            # return loss # 跳过当前 step

        # if torch.isnan(loss).any():
        #     print("Found nan in loss!", batch_idx)
        #     loss = 0 * loss
            # exit()
        # if torch.isnan(loss).any():
        #     # print("Found nan in loss!", input)
        #     print("Found NaN in loss, setting loss to 0!")
        #     loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        #     # exit()
        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_clip_loss", float(loss_clip.detach()), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_dist_loss", float(loss_dist.detach()), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_dg_loss", float(loss_dg.detach()), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        atoms = batch['complex_wt']['pos_atoms']
        mask_atoms = batch['complex_wt']['mask_atoms']
        distance_map = torch.linalg.norm(atoms[:, :, None, :, None, :]- atoms[:, None, :, None, :, :], dim=-1, ord=2).reshape(batch['size'], atoms.shape[1], atoms.shape[1], -1)
        mask = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(batch['size'], atoms.shape[1], atoms.shape[1], -1)
        distance_map[~mask] = torch.inf
        y_dist = torch.min(distance_map, dim=-1)[0]
        y_dg = batch['labels'].squeeze(1)
        res_identifier = batch['complex_wt']['group_id'] - 1
        res_identifier[res_identifier < 0] = 0
        y_clip = torch.arange(batch['size'], dtype=torch.long, device=y_dist.device)
        pred_dg, pred_dist, pred_clip = self.model(batch, stage='pretune', need_mask=True)
        dg_mask = y_dg != 0
        if dg_mask.sum() == 0:
            loss_dg = F.mse_loss(pred_dg, y_dg) * 0
        else:
            loss_dg = F.mse_loss(pred_dg, y_dg, reduction = 'none')
            loss_dg = (loss_dg * dg_mask).sum() / (dg_mask.sum() + 1e-5)
        loss_clip, loss_dist = self.cal_loss(pred_clip, y_clip, pred_dist, y_dist, res_identifier)
        val_loss = loss_clip + loss_dist + 10 * loss_dg
        skip_step = torch.tensor(int(torch.isnan(val_loss).any()), device=self.device)
        # 同步所有卡的 skip_step，最大值决定是否跳过
        skip_step = self.all_gather(skip_step).max()  # 使用 all_gather 确保所有卡同步

        if skip_step.item() == 1:
            print(f"Found NaN in loss at batch {batch_idx}, skipping this batch.")
            val_loss = torch.nan_to_num(val_loss, nan=0.0)
            loss_clip = torch.nan_to_num(loss_clip, nan=0.0)
            loss_dist = torch.nan_to_num(loss_dist, nan=0.0)
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=batch['size'], mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_clip_loss", float(loss_clip.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_dist_loss", float(loss_dist.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_dg_loss", float(loss_dg.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        res = {"val_loss_step": float(val_loss.detach()),"val_clip_loss": float(loss_clip.detach()), "val_dist_loss": float(loss_dist.detach())}
        self.res = [res]
        return val_loss
    
    def on_validation_epoch_end(self):
        val_loss = self.scalar_accum.get_average('val_loss')
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()
        
    def test_step(self, batch, batch_idx):
        atoms = batch['complex_wt']['pos_atoms']
        mask_atoms = batch['complex_wt']['mask_atoms']
        distance_map = torch.linalg.norm(atoms[:, :, None, :, None, :]- atoms[:, None, :, None, :, :], dim=-1, ord=2).reshape(batch['size'], atoms.shape[1], atoms.shape[1], -1)
        mask = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(batch['size'], atoms.shape[1], atoms.shape[1], -1)
        distance_map[~mask] = torch.inf
        y_dist = torch.min(distance_map, dim=-1)[0]
        y_dg = batch['labels'].squeeze(1)
        res_identifier = batch['complex_wt']['group_id'] - 1
        res_identifier[res_identifier < 0] = 0
        y_clip = torch.arange(batch['size'], dtype=torch.long, device=y_dist.device)
        pred_dg, pred_dist, pred_clip = self.model(batch, stage='pretune', need_mask=True)
        loss_dg = F.mse_loss(pred_dg, y_dg)
        loss_clip, loss_dist = self.cal_loss(pred_clip, y_clip, pred_dist, y_dist, res_identifier)
        test_loss = loss_clip + loss_dist
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=batch['size'], mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_clip_loss", float(loss_clip.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_dist_loss", float(loss_dist.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log("test_dg_loss", float(loss_dg.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        res = {"test_loss_step": float(test_loss.detach()),"test_clip_loss": float(loss_clip.detach()), "test_dist_loss": float(loss_dist.detach()), "test_dg_loss": float(loss_dg.detach())}
        self.res = [res]
        return test_loss

    def on_test_epoch_end(self):
        test_loss = self.scalar_accum.get_average('loss')
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
        return test_loss

