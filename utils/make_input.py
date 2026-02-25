from .tools import ProteinTools
from pathlib import Path
import shutil

def make_input(path_pdb, path_mutation, output_dir, mut_name, folding, folder_path, return_json=False):
    tools = ProteinTools()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_wt = output_dir / 'wt.pdb'
    path_mut = output_dir / mut_name
    path_mut_list = output_dir / 'mutation.txt'
    # print("Paths:", path_pdb, path_wt)
    shutil.copy(path_pdb, path_wt)
    shutil.copy(path_mutation, path_mut_list)
    if folding == 'evoef2':
        tools.evoef2(path_wt, path_mut_list, path_mut, path_bin=folder_path)
    else:
        print("Unimplemented Folding Method!")
        raise NotImplementedError
    record = {
        'path_wt': str(path_wt.name),
        'path_mut': str(path_mut.name),
        'mutation': path_mut_list.read_text().strip(';'),
    }
    if return_json:
        return record
    else:
        return None
