import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models.encoders.unibind import MonoRegularLayer
from models import ModelRegister
# from models.encoders.prompt_DDG.model import Codebook
# from models.encoders.RDE import aggregate_sidechain_accuracy, make_sidechain_accuracy_table_image
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
        # if self.num_classes > 1:
        #     loss_reg = self._get_reg_loss(pred_dict['y_pred'], y, mask)
        #     loss = loss + loss_reg

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

        # if self.num_classes > 1:
        #     loss_reg = self._get_reg_loss(pred_dict['y_pred'], y, mask)
        #     val_loss = val_loss + loss_reg

        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=batch['size'], mode='mean')
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
        # print("Validation:", results)
        res_list = []
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
        for i, l_type in enumerate(self.data_args.loss_types):
            if l_type == 'regression':
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
                
            elif l_type == 'binary':
                y_pred_i = np.array(results['y_pred_{}'.format(i)])
                y_true_i = np.array(results['y_true_{}'.format(i)])
                y_mask_i = np.array(results[f'y_mask_{i}'])
                if y_mask_i.sum() == 0:
                    continue
                else:
                    y_pred_i = y_pred_i[y_mask_i == 1]
                    y_true_i = y_true_i[y_mask_i == 1]
                acc_all = cal_accuracy(y_pred_i, y_true_i)
                auc_all = cal_auc(y_pred_i, y_true_i)
                precision_all = cal_precision(y_pred_i, y_true_i)
                recall_all = cal_recall(y_pred_i, y_true_i)
                acc_pc, precision_pc, recall_pc = per_complex_acc(results, pred_attr='y_pred_{}'.format(i), true_attr='y_true_{}'.format(i))
                print(f'[All_Task_{i}] ACC {acc_all:.6f} PRECISION {precision_all:.6f} RECALL {recall_all:.6f}')
                print(f'[PC_Task_{i}]  ACC {acc_pc:.6f} PRECISION {precision_pc:.6f} RECALL {recall_pc:.6f}')

                self.log(f'val/all_acc_{i}', acc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                # self.log(f'val/all_auc_{i}', auc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/all_precision_{i}', precision_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/all_recall_{i}', recall_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_acc_{i}', acc_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                # self.log(f'val/pc_auc_{i}', auc_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_precision_{i}', precision_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'val/pc_recall_{i}', recall_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                res = {"acc": acc_all, "auc": auc_all, "precision": precision_all, "recall": recall_all, "pc acc": acc_pc, "pc presicion": precision_pc, "pc recall": recall_pc, "pc auc": auc_pc}
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
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=batch['size'], mode='mean')
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
            if l_type == 'regression':
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
            elif l_type == 'binary':
                y_pred_i = np.array(results[f'y_pred_{i}'])
                y_true_i = np.array(results[f'y_true_{i}'])
                y_mask_i = np.array(results[f'y_mask_{i}'])
                if y_mask_i.sum() == 0:
                    continue
                else:
                    y_pred_i = y_pred_i[y_mask_i == 1]
                    y_true_i = y_true_i[y_mask_i == 1]
                acc_all = cal_accuracy(y_pred_i, y_true_i)
                auc_all = cal_auc(y_pred_i, y_true_i)
                precision_all = cal_precision(y_pred_i, y_true_i)
                recall_all = cal_recall(y_pred_i, y_true_i)
                acc_pc, auc_pc, precision_pc, recall_pc = per_complex_acc(results, pred_attr='y_pred_{}'.format(i), true_attr='y_true_{}'.format(i))
                print(f'[All_Task_{i}] ACC {acc_all:.6f} PRECISION {precision_all:.6f} RECALL {recall_all:.6f}')
                print(f'[PC_Task_{i}]  ACC {acc_pc:.6f} PRECISION {precision_pc:.6f} RECALL {recall_pc:.6f}')

                self.log(f'test/all_acc_{i}', acc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                # self.log(f'test/all_auc_{i}', auc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_precision_{i}', precision_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/all_recall_{i}', recall_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_acc_{i}', acc_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                # self.log(f'tset/pc_auc_{i}', auc_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_precision_{i}', precision_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                self.log(f'test/pc_recall_{i}', recall_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
                res = {"acc": acc_all, "auc": auc_all, "precision": precision_all, "recall": recall_all, "pc acc": acc_pc, "pc presicion": precision_pc, "pc recall": recall_pc, "pc auc": auc_pc}
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