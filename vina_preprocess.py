import numpy as np
import pandas as pd
import sys

from rdkit import Chem
from src.utils import disable_rdkit_logging
from rdkit import Chem

np.random.seed(0)

disable_rdkit_logging()

gen_smi_metric_file = sys.argv[1]
gen_smi_vina_file = sys.argv[2]

data = []
with open(gen_smi_metric_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split(' ')
        data.append({
            'scaffold': parts[0],
            'true_molecule': parts[1],
            'pred_molecule': parts[2],
            'pred_rgroup': parts[3] if len(parts) > 4 else '',
            'protein_filename': parts[4] if len(parts) > 4 else parts[3]
        })

data2 = []
with open(gen_smi_vina_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split(' ')
        data2.append({
            'scaffold': parts[0],
            'true_molecule': parts[1],
            'pred_molecule': parts[2],
            'pred_rgroup': parts[3] if len(parts) > 4 else '',
            'protein_filename': parts[4] if len(parts) > 4 else parts[3]
        })

def is_valid(pred_mol_smiles, scaf_smiles):
    if pred_mol_smiles == '':
        return False
    if pred_mol_smiles == scaf_smiles:
        return False
    pred_mol = Chem.MolFromSmiles(pred_mol_smiles)
    scaf = Chem.MolFromSmiles(scaf_smiles)
    if scaf is None:
        scaf = Chem.MolFromSmiles(scaf_smiles, sanitize=False)
    if pred_mol is None:
        pred_mol = Chem.MolFromSmiles(pred_mol_smiles, sanitize=False)
        if pred_mol is None:
            return False
    if len(pred_mol.GetSubstructMatch(scaf)) != scaf.GetNumAtoms():
        return False
    return True

for (obj, obj2) in zip(data, data2):
    valid = is_valid(obj['pred_molecule'], obj['scaffold'])
    obj['valid'] = valid
    if valid is False:
        obj2['pred_molecule'] = obj2['scaffold']


# ---------------------------- Saving -------------------------------- #

out_path = gen_smi_vina_file[:-3] + 'csv'
table = pd.DataFrame(data2)
table.to_csv(out_path, index=False)

