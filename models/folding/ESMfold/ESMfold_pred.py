import os
import torch
import esm
import typing as T
import numpy as np
from Bio import SeqIO
import argparse
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def convert_outputs_to_pdb(output: T.Dict, len_chain: T.List[int], chain_names: T.List[str]) -> T.List[str]:
    import warnings
    import string

    warnings.simplefilter("ignore")

    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    chain_index_temp = np.ones(len(final_atom_positions[0]), dtype=int)
    
    for i in range(len(len_chain)):
        if i == 0:
            start_ids = 0
            end_ids = len_chain[0]
            chain_index_temp[start_ids:end_ids] *= string.ascii_letters.index(chain_names[i])
        elif i > 0:
            start_ids = int(np.sum(len_chain[0:(i)])) + 25 * len(len_chain[0:(i)])
            end_ids = int(np.sum(len_chain[0:(i+1)])) + 25 * len(len_chain[0:(i)])
            chain_index_temp[start_ids:end_ids] *= string.ascii_letters.index(chain_names[i])

    output = {k: v.to("cpu").numpy() for k, v in output.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = output["atom37_atom_exists"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=output["plddt"][i],
            chain_index=chain_index_temp, 
        )
        pdbs.append(to_pdb(pred))
    return pdbs[0]

def main(args):
    linker = 'G' * 25
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained("./esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("./esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device)


    records = list(SeqIO.parse(args.input_file, "fasta"))

    seqs = [str(record.seq) for record in records]
    seq_names = [str(record.description) for record in records]

    chain_name = []
    all_seqs = []
    i = 0
    for each_name in seq_names:
        if 'Chains' not in each_name and 'Chain' in each_name:
            temp_name = each_name.split('|')[1].split('Chain ')[1]
            chain_name.append(temp_name[0])
            all_seqs.append(seqs[i])
        elif 'Chains' in each_name:
            temp_name = each_name.split('|')[1].split('Chains ')[1].split(',')
            for each_temp_name in temp_name:
                chain_name.append(each_temp_name.strip()[0])
                all_seqs.append(seqs[i])
        else:
            raise ValueError(f"Unexpected format in sequence name: {each_name}")
        i += 1

    temp_seqs = all_seqs
    homodimer_sequence = linker.join(temp_seqs)

    if len(homodimer_sequence) > 700:
        model.trunk.set_chunk_size(64)
    else:
        model.trunk.set_chunk_size(128)

    tokenized_homodimer = tokenizer([homodimer_sequence], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        position_ids = torch.arange(len(homodimer_sequence), dtype=torch.long)

    tokenized_homodimer['position_ids'] = position_ids.unsqueeze(0).to(device)
    tokenized_homodimer = {key: tensor.to(device) for key, tensor in tokenized_homodimer.items()}

    with torch.no_grad():
        output = model(**tokenized_homodimer)

    linker_mask = []

    seqs_len = []
    for i in range(len(temp_seqs)-1):
        linker_mask += [1] * len(temp_seqs[i])
        linker_mask += [0] * len(linker)
        seqs_len.append(len(temp_seqs[i]))
        
    linker_mask += [1] * len(temp_seqs[-1])
    seqs_len.append(len(temp_seqs[-1]))

    linker_mask = torch.tensor(linker_mask)[None, :, None].to(device)

    output['atom37_atom_exists'] = output['atom37_atom_exists'] * linker_mask
    print("Info:", seqs_len, chain_name)
    pdb = convert_outputs_to_pdb(output, seqs_len, chain_name)

    if torch.max(output['plddt']) <= 1.0:
        vmin = 0.5
        vmax = 0.95
    else:
        vmin = 50
        vmax = 95

    with open(args.output_file, "w") as f:
        f.write("".join(pdb))

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein structure prediction with ESMFold.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output PDB file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on, e.g., 'cpu' or 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=26018, help="Set random seed for ESMfold")
    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)
    main(args)
