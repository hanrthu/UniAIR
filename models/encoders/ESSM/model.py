import torch.nn as nn
import torch
import esm
from models.encoders.ESSM.encoders.pair import ResiduePairEncoder
from models.register import ModelRegister
from models.encoders.ESSM.components.coformer import CoFormer
import torch.nn.functional as F
import random
from data.transforms.essm import PreESSMTransform
from models.encoders.ESSM.lora_tune import LoRAESM, ESMConfig
from peft import (
    LoraConfig,
    get_peft_model,
)
SUPER_CPLX_IDX, SUPER_CHAIN_IDX = 21, -1
SUPER_REC_IDX, SUPER_LIG_IDX = 22, 23

R = ModelRegister()

def load_esm(esm_type):
    if esm_type == '650M':
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    elif esm_type == '3B':
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_type == '15B':
        model, _ = esm.pretrained.esm2_t48_15B_UR50D()
    elif esm_type == '150M':
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    elif esm_type == '35M':
        model, _ = esm.pretrained.esm2_t12_35M_UR50D()
    elif esm_type == '8M':
        model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise NotImplementedError
    feat_size = model.embed_dim
    return model, feat_size

def cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx):
    # print("Input shape:", prot_embedding.shape, na_embedding.shape)
    # result = prot_embedding.new_full([len(prot_embedding), seq_len, prot_embedding.shape[-1]], 0) # (N, L, E)
    new_complexes = []
    masks = []
    for i in range(len(prot_embedding)):
        item_prot_embed = prot_embedding[i]
        item_prot_mask = prot_mask[i]
        item_na_embed = na_embedding[i]
        item_na_mask = na_mask[i]
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad)
    result = torch.stack(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks

def segment_cat_pad(prot_embedding, prot_chains, prot_mask, max_len, patch_idx=None):
    cum_prot = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(prot_chains), dim=0)]).int()
    new_complexes = []
    masks = []
    for i, (s_prot, e_prot) in enumerate(zip(cum_prot[:-1], cum_prot[1:])):
        item_prot_embed = prot_embedding[s_prot:e_prot].reshape((-1, prot_embedding.shape[-1]))
        item_prot_mask = prot_mask[s_prot:e_prot].reshape(-1)
        indices = torch.nonzero(item_prot_mask).flatten()
        selected = torch.index_select(item_prot_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        # # selected_pad = torch.cat([selected, torch.zeros((seq_len-len(selected), prot_embedding.shape[-1]), device=selected.device)], dim=0)
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad.unsqueeze(0))
    result = torch.cat(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    # print("Result shape:", result)
    return result, masks

def segment_pool(input, chains, mask, pooling):
    # input shape [N', L, E], mask_shape [N', L]
    result = input.new_full([len(chains), input.shape[-1]], 0) # (N, E)
    mask_result = mask.new_full([len(chains), 1], 0) #(N, 1)
    input_flattened = input.reshape((-1, input.shape[-1])) #(N'*L, E)
    mask_flattened = mask.reshape((-1, 1)) #(N'*L, 1)
    # print("Shapes:", result.shape, mask_result.shape, input_flattened.shape, mask_flattened.shape)
    # segment_id shape (N', )
    segment_id = torch.tensor(sum([[i] * chain for i, chain in enumerate(chains)], start=[]), device=result.device, dtype=torch.int64)
    segment_id = segment_id.repeat_interleave(input.shape[1]) #(N'*L)
    result.scatter_add_(0, segment_id.unsqueeze(1).expand_as(input_flattened), input_flattened*mask_flattened)
    mask_result.scatter_add_(0, segment_id.unsqueeze(1), mask_flattened)
    mask_result.reshape((-1, ))
    
    if pooling == 'mean':
        result = result / (mask_result + 1e-10)
    
    return result

@R.register('essm')
class ESSM(nn.Module):
    def __init__(self, 
                 esm_type='650M',
                 output_dim=1,
                 pair_dim=40,
                 fix_lms=True,
                 lora_tune=False,
                 lora_rank=16,
                 lora_alpha=32,
                 representation_layer=33,
                 dist_dim=40,
                 pre_transform_cfg={},
                 **kwargs
                 ):
        super(ESSM, self).__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.feat_size = esm_feat_size
        self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=5)  # N, CA, C, O, CB
        self.c_former = CoFormer(**kwargs['coformer'])
        self.representation_layer = representation_layer
        self.proj = 0
        self.complex_dim = kwargs['coformer']['embed_dim']
        self.proj_cplx = nn.Linear(self.feat_size, self.complex_dim)
        self.transform = PreESSMTransform(**pre_transform_cfg)
        if lora_tune:
            import re
            pattern = r'\((\w+)\): Linear'
            esm_linear_layers = re.findall(pattern, str(self.esm.modules))
            esm_linear_modules = list(set(esm_linear_layers))
            print("In esm:", esm_linear_modules)
            print("Getting Lora Models...")
            # copied from LongLoRA
            esm_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                target_modules=esm_linear_modules,
                lora_alpha=lora_alpha
            )
            esm_config = ESMConfig()
            self.esm = LoRAESM(self.esm, esm_config)
            self.esm = get_peft_model(self.esm, esm_lora_config)
            print("Get ESM DONE!!!!!")
            self.esm.print_trainable_parameters()
        elif fix_lms:
            for p in self.esm.parameters():
                p.requires_grad_(False)
        
        # Special token for token pooling
        self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
        nn.init.normal_(self.complex_embedding)
        
        if pair_dim != self.complex_dim:
            self.z_proj = nn.Linear(pair_dim, self.complex_dim)
            
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
    
    def _forward(self, input):
        prot_input = input['prot_seqs']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        data = input['complex_wt']
        data = self.transform(data)

        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]

        prot_embedding = prot_embedding.float()
        max_len = data['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer
        if 'patch_idx' in data:
            patch_idx = data['patch_idx']
        else:
            patch_idx = None
        
        # Cat pad to generate batched embedding
        out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, max_len, patch_idx)
        assert out_embedding.shape[0] == data['aa'].shape[0]

        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks
        
        aa=data['aa']
        res_nb=data['res_nb']
        chain_nb=data['chain_nb']
        pos_atoms=data['pos_atoms']
        mask_atoms=data['mask_atoms']
        
        # Token pooling
        mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
        cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
        
        out_embedding = torch.cat([cplx_embed, out_embedding], dim=1)
        key_padding_mask = torch.cat([mask_special, key_padding_mask], dim=1)
        
        cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
        aa = torch.cat([cplx_type, aa], dim=1)
        
        res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
        
        res_nb = torch.cat([res_nb_cplx, res_nb], dim=1)
        super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
        chain_nb = torch.cat([super_chain_id, chain_nb], dim=1)
        
        center_cplx = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2], 3), device=out_embedding.device, dtype=pos_atoms.dtype)
        pos_atoms = torch.cat([center_cplx, pos_atoms], dim=1)
        # noise = torch.randn_like(pos_atoms, dtype=torch.float32, device=pos_atoms.device)
        # pos_atoms += noise
        mask_atom = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2]), device=out_embedding.device, dtype=mask_atoms.dtype)
        mask_atom[:,:,0] = 1
        mask_atoms = torch.cat([mask_atom, mask_atoms], dim=1)


        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
        )
            
        return out_embedding, z, key_padding_mask
    
    def encode(self, input, stage='mutation'):
        prot_input = input['prot_seqs']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        data = input['complex_wt']
        data = self.transform(data)
        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]

        prot_embedding = prot_embedding.float()
        max_len = data['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer
        if 'patch_idx' in data:
            patch_idx = data['patch_idx']
        else:
            patch_idx = None
        
        # Cat pad to generate batched embedding
        out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, max_len, patch_idx)
        assert out_embedding.shape[0] == data['aa'].shape[0]
        return out_embedding
    
    def _encode(self, input, stage='mutation'):
        out_embedding, z, key_padding_mask = self._forward(input)
        input['prot_seqs'] = input['prot_mut']
        # input['restype'] = input['mut_restype']
        out_mut, z_mut, _ = self._forward(input)
        output_wild, z_wild, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)
        output_mut, z_mut, attn = self.c_former(out_mut, z_mut, key_padding_mask=key_padding_mask, need_attn_weights=False)
        wild_embedding = output_wild + self.z_proj(z_wild).sum(-2) * 0.001
        # Default to be token embeding
        wild_embedding = wild_embedding[:, 0, :].squeeze(1)
        mut_embedding = output_mut + self.z_proj(z_mut).sum(-2) * 0.001
        mut_embedding = mut_embedding[:, 0, :].squeeze(1)
        
        forward_embedding = wild_embedding - mut_embedding
        return forward_embedding
    
    def encode_for_unimut(self, input):
        return self._encode(input)
    
    def forward_encode(self, input):
        with torch.no_grad():
            out_embedding, z, key_padding_mask = self._forward(input)
            input['prot_seqs'] = input['prot_mut']
            # input['restype'] = input['mut_restype']
            out_mut, z_mut, _ = self._forward(input)
        output_wild, z_wild, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)
        output_mut, z_mut, attn = self.c_former(out_mut, z_mut, key_padding_mask=key_padding_mask, need_attn_weights=False)
        wild_embedding = output_wild + self.z_proj(z_wild).sum(-2) * 0.001
        # Default to be token embeding
        mut_embedding = output_mut + self.z_proj(z_mut).sum(-2) * 0.001
        embedding = wild_embedding[:, 0, :].squeeze(1) - mut_embedding[:, 0, :].squeeze(1)
        return embedding
    
    def forward_readout(self, h_in):
        forward_embedding = h_in
        inv_embedding = -h_in
        
        output_forward = self.pred_head(forward_embedding).squeeze(1)
        output_inv = self.pred_head(inv_embedding).squeeze(1)
        pred_dict = {
            'y_pred': output_forward, 
            'y_pred_inv': output_inv
        }
        return pred_dict, forward_embedding
    
    def forward(self, batch, stage='None'):
        # if stage == 'None':
        h_in = self.forward_encode(batch)
        pred_dict, _ = self.forward_readout(h_in)
        return pred_dict

    def inference(self, input, stage='mutation'):
        return self.forward(input, stage)
        

@R.register('ESSM_pretune')
class ESSM_Pretune(nn.Module):
    def __init__(self, 
                 esm_type='650M',
                 output_dim=1,
                 pair_dim=40,
                 fix_lms=True,
                 representation_layer=33,
                 dist_dim=40,
                 lora_tune=False,
                 lora_rank=16,
                 lora_alpha=32,
                 pre_transform_cfg={},
                 **kwargs
                 ):
        super(ESSM_Pretune, self).__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.feat_size = esm_feat_size
        self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=5)  # N, CA, C, O, CB
        self.c_former = CoFormer(**kwargs['coformer'])
        self.representation_layer = representation_layer
        self.proj = 0
        self.complex_dim = kwargs['coformer']['embed_dim']
        self.proj_cplx = nn.Linear(self.feat_size, self.complex_dim)
        self.transform = PreESSMTransform(**pre_transform_cfg)
        if lora_tune:
            import re
            pattern = r'\((\w+)\): Linear'
            esm_linear_layers = re.findall(pattern, str(self.esm.modules))
            esm_linear_modules = list(set(esm_linear_layers))
            print("In esm:", esm_linear_modules)
            print("Getting Lora Models...")
            # copied from LongLoRA
            esm_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                target_modules=esm_linear_modules,
                lora_alpha=lora_alpha
            )
            esm_config = ESMConfig()
            self.esm = LoRAESM(self.esm, esm_config)
            self.esm = get_peft_model(self.esm, esm_lora_config)
            print("Get ESM DONE!!!!!")
            self.esm.print_trainable_parameters()
        elif fix_lms:
            for p in self.esm.parameters():
                p.requires_grad_(False)
        
        # Special token for token pooling
        self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))

        self.lig_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
        self.rec_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
        nn.init.normal_(self.complex_embedding)
        nn.init.normal_(self.lig_embedding)
        nn.init.normal_(self.rec_embedding)
        self.mask_token = nn.Parameter(torch.randn(size=(1, pair_dim)))
        
        if pair_dim != self.complex_dim:
            self.z_proj = nn.Linear(pair_dim, self.complex_dim)
            
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
        self.dist_head = nn.Sequential(
            nn.Linear(pair_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, dist_dim)
        )
        
    
    def _forward(self, input, need_mask=False):
        prot_input = input['prot_seqs']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        data = input['complex_wt']
        data = self.transform(data)
        identifier = data['group_id'] - 1
        identifier[identifier < 0] = 0

        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]

        prot_embedding = prot_embedding.float()
        max_len = data['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer
        if 'patch_idx' in data:
            patch_idx = data['patch_idx']
        else:
            patch_idx = None
        
        # Cat pad to generate batched embedding
        out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, max_len, patch_idx)
        assert out_embedding.shape[0] == data['aa'].shape[0]

        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks
        
        aa=data['aa']
        res_nb=data['res_nb']
        chain_nb=data['chain_nb']
        pos_atoms=data['pos_atoms']
        mask_atoms=data['mask_atoms']
        
        # # Token pooling
        
        mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
        cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
        lig_embed = self.lig_embedding.repeat(len(out_embedding), 1, 1)
        rec_embed = self.rec_embedding.repeat(len(out_embedding), 1, 1)
        
        out_embedding = torch.cat([cplx_embed, lig_embed, rec_embed, out_embedding], dim=1)
        key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
        
        cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
        lig_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_LIG_IDX
        rec_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_REC_IDX
        aa = torch.cat([cplx_type, lig_type, rec_type, aa], dim=1)
        
        res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
        res_nb_lig = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 1
        res_nb_rec = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 2
        
        res_nb = torch.cat([res_nb_cplx, res_nb_lig, res_nb_rec, res_nb], dim=1)
        super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
        chain_nb = torch.cat([super_chain_id, super_chain_id, super_chain_id, chain_nb], dim=1)
        
        center_cplx = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2], 3), device=out_embedding.device, dtype=pos_atoms.dtype)
        center_lig = ((pos_atoms * (1-identifier)[:, :, None, None] * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((1-identifier[:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, pos_atoms.shape[2], 1)
        center_rec = ((pos_atoms * (identifier[:, :, None, None]) * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((identifier[:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, pos_atoms.shape[2], 1)
        # print(center_cplx.shape, center_lig.shape, center_rec.shape)
        pos_atoms = torch.cat([center_cplx, center_lig, center_rec, pos_atoms], dim=1)
        # noise = torch.randn_like(pos_atoms, dtype=torch.float32, device=pos_atoms.device)
        # pos_atoms += noise
        mask_atom = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2]), device=out_embedding.device, dtype=mask_atoms.dtype)
        mask_atom[:,:,0] = 1
        mask_atoms = torch.cat([mask_atom, mask_atom, mask_atom, mask_atoms], dim=1)

        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
        )
        if need_mask:
            # Random mask rows/columns, 50% probability to mask 15% positions, 50% probability to keep the same
            for i in range(z.shape[0]):
                to_mask = torch.rand(1).item() > 0.5
                if not to_mask:
                    continue
                valid = list(range(3, z.shape[1]))
                mask_indices = random.sample(valid, int(len(valid) * 0.15))
                z[i, mask_indices, :, :] = self.mask_token.repeat(len(mask_indices), z.shape[2], 1)
                z[i, :, mask_indices, :] = self.mask_token.repeat(z.shape[1], len(mask_indices), 1)
         
        return out_embedding, z, key_padding_mask
    
    def forward(self, input, stage='mutation', need_mask=False):
        out_embedding, z, key_padding_mask = self._forward(input, need_mask=need_mask)
        if stage == 'pretune':
            # -------------------------------------------dG prediction ----------------------------------------------     
            output, z, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)

            complex_embedding = output + self.z_proj(z).sum(-2) * 0.001
            
            #Token pooling
            complex_embedding = output[:, 0, :].squeeze(1)

            output = self.pred_head(complex_embedding)
            dG = output.squeeze(1)
            # -------------------------------------------CLIP feature generation ----------------------------------------------
            # return output
            res_identifier = input['complex_wt']['group_id'] - 1
            res_identifier[res_identifier < 0] = 0
            attn_mask = torch.ones((out_embedding.shape[0], out_embedding.shape[1], out_embedding.shape[1]), device=out_embedding.device).bool()
            
            lig_token_identifier = torch.zeros(len(out_embedding), 1, dtype=res_identifier.dtype, device=res_identifier.device)
            rec_token_identifier = torch.ones(len(out_embedding), 1, dtype=res_identifier.dtype, device=res_identifier.device)
            res_identifier = torch.cat([lig_token_identifier, rec_token_identifier, res_identifier], dim=1)
            attn_mask[:, 1:, 1:] = (res_identifier[:, :, None] == res_identifier[:, None, :])
            
            attn_mask = ~attn_mask
            # all the ones in transformer mask means ignoring, which is different from the meaning of pos_mask !!!!
            if torch.isnan(z).any():
                print("Found Nan in z!")
            output, z, _ = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False, attn_mask=attn_mask)
            
            # Output Embedding: [N, E]
            complex_embedding = output[:, 0, :].squeeze(1)
            lig_embedding = output[:, 1, :].squeeze(1)
            rec_embedding = output[:, 2, :].squeeze(1)
                    
            similarity = F.cosine_similarity(lig_embedding[:, None, :], rec_embedding[None, :, :], dim=2)

            if torch.isnan(z).any():
                print("Found Nan in z!")
            # ------------------------------------- Atom-level distance prediction -------------------------------------------
            
            output, z, _ = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False, attn_mask=None)

            if torch.isnan(z).any():
                print("Found Nan in z!")

            dist_logits = self.dist_head(z)
            dist_logits = dist_logits[:, 3:, 3:, :]
            # print("Dist_logist:", dist_logits.shape)
            # dist_prob = F.softmax(dist_logits, dim=-1)
            return dG, dist_logits, similarity
        
        
        elif stage == 'mutation':
            input['prot_seqs'] = input['prot_mut']
            input['complex_wt']['aa'] = input['complex_wt']['aa_mut']
            # input['restype'] = input['mut_restype']
            out_mut, z_mut, _ = self._forward(input)
            deep = False
            output_wild, z_wild, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)
            output_mut, z_mut, attn = self.c_former(out_mut, z_mut, key_padding_mask=key_padding_mask, need_attn_weights=False)
            wild_embedding = output_wild + self.z_proj(z_wild).sum(-2) * 0.001
            # Default to be token embeding
            wild_embedding = wild_embedding[:, 0, :].squeeze(1)
            mut_embedding = output_mut + self.z_proj(z_mut).sum(-2) * 0.001
            mut_embedding = mut_embedding[:, 0, :].squeeze(1)
            
            
            forward_embedding = wild_embedding - mut_embedding
            inv_embedding = mut_embedding - wild_embedding
            
            output_forward = self.pred_head(forward_embedding).squeeze(1)
            output_inv = self.pred_head(inv_embedding).squeeze(1)
            
            return {
                'y_pred': output_forward, 
                'y_pred_inv': output_inv
            }

    def inference(self, input, stage='mutation'):
        return self.forward(input, stage)
        
@R.register('ESM2')
class SimpleESM2(nn.Module):
    def __init__(self, 
                 esm_type='650M',
                 output_dim=1,
                 fix_lms=True,
                 representation_layer=33,
                 complex_dim=320,
                 pre_transform_cfg={},
                 **kwargs
                 ):
        super(SimpleESM2, self).__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.feat_size = esm_feat_size
        self.representation_layer = representation_layer
        self.proj = 0
        self.complex_dim = complex_dim
        self.proj_cplx = nn.Linear(self.feat_size, self.complex_dim)
        self.transform = PreESSMTransform(**pre_transform_cfg)
        if fix_lms:
            for p in self.esm.parameters():
                p.requires_grad_(False)
        
        # Special token for token pooling
        self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
        nn.init.normal_(self.complex_embedding)
        
            
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
    
    def _forward(self, input):
        prot_input = input['prot_seqs']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        data = input['complex_wt']
        data = self.transform(data)

        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]

        prot_embedding = prot_embedding.float()
        max_len = data['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer
        if 'patch_idx' in data:
            patch_idx = data['patch_idx']
        else:
            patch_idx = None
        out_embedding = segment_pool(prot_embedding, prot_chains, prot_mask, pooling='mean')
        # Cat pad to generate batched embedding
        assert out_embedding.shape[0] == data['aa'].shape[0]

        out_embedding = self.proj_cplx(out_embedding)
     
        return out_embedding
    
    def encode(self, input, stage='mutation'):
        return self._forward(input)
    
    def forward(self, input, stage='mutation'):
        out_embedding = self._forward(input)
        input['prot_seqs'] = input['prot_mut']
        # input['restype'] = input['mut_restype']
        out_mut= self._forward(input)
        
        out_forward = out_embedding - out_mut
        out_inv = out_mut - out_embedding
            
        output_forward = self.pred_head(out_forward)
        output_forward = output_forward.squeeze(1)

        output_inv = self.pred_head(out_inv)
        output_inv = output_inv.squeeze(1)
        
        return {
            'y_pred': output_forward, 
            'y_pred_inv': output_inv
        }        
        

    def inference(self, input, stage='mutation'):
        return self.forward(input, stage)
        
            
            

