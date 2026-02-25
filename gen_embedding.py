import torch
import torch.nn as nn
from models.register import ModelRegister
from pl_modules.model_module import MutationModelModule
from pl_modules.data_module import MutationDataModule
from models.encoders.LatentAdapter.transformer.transformer import Transformer
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import os
from easydict import EasyDict

def cal_embed_with_pred(dataloader, models, encoder_names, device='cpu', root_dir = None):
    embed_dict = {}
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        embed_id = batch['complex'][0] + '_' + batch['mutstr'][0] + '.pt'
        out_dir = os.path.join(root_dir, embed_id)
        if os.path.exists(out_dir):
            continue
        # print(batch)
        embed_dict[i] = {}
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for item in batch[key]:
                    if isinstance(batch[key][item], torch.Tensor):
                        batch[key][item] = batch[key][item].to(device)
        for encoder, name in zip(models, encoder_names):
            with torch.no_grad():
                encoded = encoder.forward_encode(batch)
            encoded_cpu = []
            for item in encoded:
                if isinstance(item, torch.Tensor):
                    encoded_cpu.append(item.squeeze().cpu())
                elif isinstance(item, tuple):
                    if item[1] is not None:
                        encoded_cpu.append((item[0].squeeze().cpu(), item[1].squeeze().cpu()))
                    else:
                        encoded_cpu.append((item[0].squeeze().cpu(), None))
                else:
                    encoded_cpu.append(item)
            embed_dict[i][name] = {}
            embed_dict[i][name]['real'] = encoded_cpu
            
            # predicted structures
            batch_pred = {k: v for k, v in batch.items()}
            batch_pred['complex_wt']['pos_atoms'] = batch_pred['complex_wt']['pred_pos_atoms']
            batch_pred['complex_wt']['pos_heavyatom'] = batch_pred['complex_wt']['pred_pos_heavyatom']
            if 'complex_mut' in batch_pred:
                batch_pred['complex_mut']['pos_atoms'] = batch_pred['complex_mut']['pred_pos_atoms']
                batch_pred['complex_mut']['pos_heavyatom'] = batch_pred['complex_mut']['pred_pos_heavyatom']
            with torch.no_grad():
                encoded = encoder.forward_encode(batch_pred)
            encoded_cpu = []
            
            for item in encoded:
                if isinstance(item, torch.Tensor):
                    encoded_cpu.append(item.squeeze().cpu())
                elif isinstance(item, tuple):
                    if item[1] is not None:
                        encoded_cpu.append((item[0].squeeze().cpu(), item[1].squeeze().cpu()))
                    else:
                        encoded_cpu.append((item[0].squeeze().cpu(), None))
                else:
                    encoded_cpu.append(item)
            embed_dict[i][name]['pred'] = encoded_cpu
        embed_dict[i]['labels'] = batch['labels'].squeeze().cpu()
        embed_dict[i]['label_mask'] = batch['label_mask'].squeeze().cpu()
        embed_dict[i]['complex'] = batch['complex'][0]
        embed_dict[i]['mutstr'] = batch['mutstr'][0]
        embed_dict[i]['id'] = embed_id
        # print("Labels:", batch['labels'].shape, batch['label_mask'].shape)
        # break
        torch.save(embed_dict[i], out_dir)
    return embed_dict
def cal_embed(dataloader_gen, dataloader_des, models, encoder_names, device='cpu', root_dir=None):
    # embed_dict = {}

    for i, (batch_gen, batch_des) in tqdm(enumerate(zip(dataloader_gen, dataloader_des)), total=len(dataloader_gen)):
        # breakpoint()
        batch = batch_gen
        # print(batch)
        assert batch_gen['complex'][0] == batch_des['complex'][0]
        assert batch_des['complex'][0] == batch_des['complex'][0]
        embed_id = batch['complex'][0] + '_' + batch['mutstr'][0] + '.pt'
        out_dir = os.path.join(root_dir, embed_id)
        if os.path.exists(out_dir):
            continue
        embed_dict = {}
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for item in batch[key]:
                    if isinstance(batch[key][item], torch.Tensor):
                        batch[key][item] = batch[key][item].to(device)
        for key in batch_des:
            if isinstance(batch_des[key], torch.Tensor):
                batch_des[key] = batch_des[key].to(device)
            elif isinstance(batch_des[key], dict):
                for item in batch_des[key]:
                    if isinstance(batch_des[key][item], torch.Tensor):
                        batch_des[key][item] = batch_des[key][item].to(device)
        for encoder, name in zip(models, encoder_names):
            if name != 'gearbind':
                with torch.no_grad():
                    encoded = encoder.forward_encode(batch)
            else:
                with torch.no_grad():
                    encoded = encoder.forward_encode(batch_des)
            encoded_cpu = []
            # for item in encoded:
            #     if isinstance(item, torch.Tensor):
            #         encoded_cpu.append(item.squeeze().cpu())
            #     elif isinstance(item, tuple):
            #         if item[1] is not None:
            #             encoded_cpu.append((item[0].squeeze().cpu(), item[1].squeeze().cpu()))
            #         else:
            #             encoded_cpu.append((item[0].squeeze().cpu(), None))
            #     else:
            #         encoded_cpu.append(item)
            # Only for Light ens
            if name == 'gearbind':
                embedding = {}
                embedding['ab'] = torch.cat([encoded[0][1], encoded[0][0]], dim=-1).squeeze()
                embedding['ba'] = torch.cat([encoded[0][0], encoded[0][1]], dim=-1).squeeze()
            elif name == 'rde':
                h_wt, h_mt = encoded[0]
                H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
                embedding = H_mt - H_wt
            elif name == 'ppiformer':
                y_logits = encoder.classifier(encoded[0][0])
                y_proba = y_logits.softmax(dim=-1)
                wt_proba = torch.gather(y_proba, 2, encoded[1].unsqueeze(-1))
                mt_proba = torch.gather(y_proba, 2, encoded[2].unsqueeze(-1))
                log_odds = wt_proba.log() - mt_proba.log()
                embedding = (log_odds * encoded[3].unsqueeze(-1)).squeeze(dim=-1)
            else:
                embedding = encoded[0][0][:, 0, :].squeeze(1) - encoded[0][1][:, 0, :].squeeze(1)
            embed_dict[name] = embedding
        embed_dict['labels'] = batch['labels'].squeeze().cpu()
        embed_dict['label_mask'] = batch['label_mask'].squeeze().cpu()
        embed_id = batch['complex'][0] + '_' + batch['mutstr'][0]
        embed_dict['complex'] = batch['complex'][0]
        embed_dict['mutstr'] = batch['mutstr'][0]
        embed_dict['id'] = embed_id
        valid_list = ['pos_atoms', 'mask_atoms', 'bfactor_atoms', 'aa', 'aa_mut', 'mut_flag' , 'chi', 'chi_mask', 'phi', 'phi_mask', 'psi', 'psi_mask', 'res_nb', 'chain_nb']
        complex_wt = {}
        for key in batch['complex_wt']:
            item = batch['complex_wt'][key]
            if key not in valid_list:
                continue
            complex_wt[key] = item.squeeze().cpu()
        embed_dict['complex_wt'] = complex_wt
        # breakpoint()
        # print("Labels:", batch['labels'].shape, batch['label_mask'].shape)
        # break
        torch.save(embed_dict, out_dir)
    return embed_dict

model_yaml = './config/models/train/UniPPI_meanens.yaml'
dataset_yaml_des = './config/datasets/descriminative/Gearbind/SKEMPIv2.yaml'
# dataset_yaml = './config/datasets/transfer/SKEMPIv2.yaml'
dataset_yaml_gen = './config/datasets/generative/RDE/SKEMPIv2.yaml'
# out_train_dir = './datasets/SKEMPIv2_light_transfer_openfold/train'
# out_val_dir = './datasets/SKEMPIv2_light_transfer_openfold/val'
out_train_dir = './datasets/SKEMPIv2_light_new4/train'
out_val_dir = './datasets/SKEMPIv2_light_new4/val'
os.makedirs(out_train_dir, exist_ok=True)
os.makedirs(out_val_dir, exist_ok=True)
with open(model_yaml, 'r') as f:
    content = f.read()
    config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))

with open(dataset_yaml_gen, 'r') as f:
    content = f.read()
    dataset_args_gen = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
    dataset_args_gen.batch_size = 1

with open(dataset_yaml_des, 'r') as f:
    content = f.read()
    dataset_args_des = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
    dataset_args_des.batch_size = 1
    
encoder_dirs = config_dict['model']['model_dirs']
encoder_names = ['rde', 'ppiformer', 'lmbind', 'gearbind']

folds = [0, 1, 2]
device = 'cuda:1'
embed_dict = {}
for fold in folds: 
    os.makedirs(os.path.join(out_train_dir, 'fold_' + str(fold)), exist_ok=True)
    os.makedirs(os.path.join(out_val_dir, 'fold_' + str(fold)), exist_ok=True)
    models = []
    for name in encoder_names:
        encoder = MutationModelModule.load_from_checkpoint(encoder_dirs[name][fold], map_location=torch.device(device)).model
        models.append(encoder)
    data_module_gen = MutationDataModule(dataset_args=dataset_args_gen, **dataset_args_gen, col_group=f'fold_{fold}')
    data_module_gen.setup()
    data_module_des = MutationDataModule(dataset_args=dataset_args_des, **dataset_args_des, col_group=f'fold_{fold}')
    data_module_des.setup()

    trainloader_gen = data_module_gen.train_dataloader(shuffle=False)
    valloader_gen = data_module_gen.val_dataloader()
    trainloader_des = data_module_des.train_dataloader(shuffle=False)
    valloader_des = data_module_des.val_dataloader()

    for batch in trainloader_gen:
        if 'pred_pos_atoms' in batch['complex_wt']:
            cal = cal_embed_with_pred
            break
        else:
            cal = cal_embed
            break
    train_embed = cal(trainloader_gen, trainloader_des, models, encoder_names, device=device, root_dir=out_train_dir + '/fold_' +str(fold))
    val_embed = cal(valloader_gen, valloader_des, models, encoder_names, device=device, root_dir = out_val_dir + '/fold_' + str(fold))    
    
    # for key in val_embed:
    #     embed_name = val_embed[key]['id']
    #     out_dir = os.path.join(out_val_dir, 'fold_' + str(fold), embed_name + '.pt')
    #     torch.save(val_embed[key], out_dir)
        
    # print(embed_dict)
# torch.save(embed_dict, 'SKEMPIv2_embedding_generative_with_label.pt')
    # break
    # break
    # print(fold)
# print(encoder)