import fire
from pathlib import Path
import shutil
from subprocess import getoutput

class ProteinTools(object):
    def __init__(self):
        super(ProteinTools, self).__init__()

    def evoef2(self, path_pdb, path_mutation, path_output, path_bin='EvoEF2'):
        path_pdb = Path(path_pdb)
        path_output = Path(path_output)
        path_bin = Path(path_bin)
        path_output.parent.mkdir(parents=True, exist_ok=True)

        path_pdb_temp = path_bin.parent / 'input.pdb'
        path_mut_temp = path_bin.parent / 'mutation.txt'
        path_output_temp = path_bin.parent / 'input_Model_0001.pdb'

        shutil.copy(path_pdb, path_pdb_temp)
        shutil.copy(path_mutation, path_mut_temp)
        cmd = f'cd {path_bin.parent}; ./{path_bin.name} --command=BuildMutant --pdb input.pdb --mutant_file mutation.txt'
        msg = getoutput(cmd)
        # print(msg)
        shutil.copy(path_output_temp, path_output)
        return msg

    def esmfold(self, path_pdb, path_mutation, path_output, path_bin='ESMfold'):
        pass

    def openfold(self, path_pdb, path_mutation, path_output, path_bin='openfold'):
        pass