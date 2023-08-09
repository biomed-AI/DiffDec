import numpy as np
import pandas as pd
import sys

from rdkit import Chem
from rdkit.Chem import DataStructs
from src.utils import disable_rdkit_logging
from rdkit import Chem

np.random.seed(0)

disable_rdkit_logging()

gen_smi_file = sys.argv[1]

data = []
true_mol_set = set()
protein_filename_set = set()
with open(gen_smi_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split(' ')
        data.append({
            'scaffold': parts[0],
            'true_molecule': parts[1],
            'pred_molecule': parts[2],
            'pred_rgroup': parts[3] if len(parts) > 4 else '',
            'protein_filename': parts[4] if len(parts) > 4 else parts[3]
        })
        true_mol_set.add(data[-1]['true_molecule'])
        protein_filename_set.add(data[-1]['protein_filename'])

true_mol_list = list(true_mol_set)
protein_filename_list = list(protein_filename_set)

summary = {}

# -------------- Validity -------------- #
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

val_dict = {}
for i in range(len(protein_filename_list)):
    val_dict[protein_filename_list[i]] = []
tmp_dict = dict()
for obj in data:
    valid = is_valid(obj['pred_molecule'], obj['scaffold'])
    obj['valid'] = valid

    protein_filename = obj['protein_filename']
    true_scaf = obj['scaffold']
    key = f'{protein_filename}|{true_scaf}'
    tmp_dict.setdefault(key, []).append(valid)

for k, samples in tmp_dict.items():
    protein_filename = k.split('|')[0]
    val_dict[protein_filename].append(sum(samples) / len(samples))

avg_tmp = []
for k, v in val_dict.items():
    if len(v) == 0:
        continue
    avg_tmp.append(sum(v) / len(v))
validity = sum(avg_tmp) / len(avg_tmp) * 100
print(f'Validity: {validity:.3f}%')

summary['validity'] = validity

# -------------- Uniqueness -------------- #
uni_dict = {}
for i in range(len(protein_filename_list)):
    uni_dict[protein_filename_list[i]] = []
tmp_dict = dict()
for obj in data:
    if not obj['valid']:
        continue
    protein_filename = obj['protein_filename']
    true_mol = obj['true_molecule']
    true_scaf = obj['scaffold']
    key = f'{protein_filename}|{true_scaf}'
    tmp_dict.setdefault(key, []).append(obj['pred_molecule'])

unique_cnt = 0
for k, samples in tmp_dict.items():
    protein_filename = k.split('|')[0]
    uni_dict[protein_filename].append(len(set(samples)) / 100)
    unique_cnt += len(set(samples))

avg_tmp = []
for k, v in uni_dict.items():
    if len(v) == 0:
        continue
    avg_tmp.append(sum(v) / len(v))
uniqueness = sum(avg_tmp) / len(avg_tmp) * 100
print(f'Uniqueness: {uniqueness:.3f}%')
summary['uniqueness'] = uniqueness

# ---------------------------- Similarity -------------------------------- #
sim_dict = {}
for i in range(len(protein_filename_list)):
    sim_dict[protein_filename_list[i]] = []
for obj in data:
    if not obj['valid']:
        # obj['sim'] = None
        continue
    
    pred_mol = Chem.MolFromSmiles(obj['pred_molecule'])
    if pred_mol is None:
        pred_mol = Chem.MolFromSmiles(obj['pred_molecule'], sanitize=False)
    pred_mol = Chem.RDKFingerprint(pred_mol)
    true_mol = Chem.MolFromSmiles(obj['true_molecule'])
    if true_mol is None:
        true_mol = Chem.MolFromSmiles(obj['true_molecule'], sanitize=False)
    true_mol = Chem.RDKFingerprint(true_mol)
    sim = DataStructs.FingerprintSimilarity(pred_mol, true_mol)

    sim_dict[obj['protein_filename']].append(sim)

avg_tmp = []
for k, v in sim_dict.items():
    if len(v) == 0:
        continue
    avg_tmp.append(sum(v) / len(v))
print(f'Similarity: {sum(avg_tmp) / len(avg_tmp):.3f}')
# summary['sim'] = sum(avg_tmp) / len(avg_tmp)

# ----------------- Recovery ---------------- #
r_scaf_dict = {}
r_recovered_dict = {}

for obj in data:
    if not obj['valid']:
        obj['recovered'] = False
        try:
            scaf_mol = Chem.MolFromSmiles(obj['scaffold'])
            Chem.RemoveStereochemistry(scaf_mol)
            scaf_smi = Chem.MolToSmiles(Chem.RemoveHs(scaf_mol))
        except:
            scaf_mol = Chem.MolFromSmiles(obj['scaffold'], sanitize=False)
            Chem.RemoveStereochemistry(scaf_mol)
            scaf_smi = Chem.MolToSmiles(Chem.RemoveHs(scaf_mol, sanitize=False))

        key_ = obj['protein_filename']
        if key_ not in r_scaf_dict:
            r_scaf_dict[key_] = set()
            r_recovered_dict[key_] = set()
        r_scaf_dict[key_].add(scaf_smi)
        continue

    try:
        true_mol = Chem.MolFromSmiles(obj['true_molecule'])
        Chem.RemoveStereochemistry(true_mol)
        true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(true_mol))
    except:
        true_mol = Chem.MolFromSmiles(obj['true_molecule'], sanitize=False)
        Chem.RemoveStereochemistry(true_mol)
        true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(true_mol, sanitize=False))

    try:
        scaf_mol = Chem.MolFromSmiles(obj['scaffold'])
        Chem.RemoveStereochemistry(scaf_mol)
        scaf_smi = Chem.MolToSmiles(Chem.RemoveHs(scaf_mol))
    except:
        scaf_mol = Chem.MolFromSmiles(obj['scaffold'], sanitize=False)
        Chem.RemoveStereochemistry(scaf_mol)
        scaf_smi = Chem.MolToSmiles(Chem.RemoveHs(scaf_mol, sanitize=False))

    try:
        pred_mol = Chem.MolFromSmiles(obj['pred_molecule'])
        Chem.RemoveStereochemistry(pred_mol)
        pred_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(pred_mol))
    except:
        pred_mol = Chem.MolFromSmiles(obj['pred_molecule'], sanitize=False)
        Chem.RemoveStereochemistry(pred_mol)
        pred_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(pred_mol, sanitize=False))

    key_ = obj['protein_filename']
    
    if key_ not in r_scaf_dict:
        r_scaf_dict[key_] = set()
        r_recovered_dict[key_] = set()

    recovered = true_mol_smi == pred_mol_smi
    obj['recovered'] = recovered
    if recovered:
        r_recovered_dict[key_].add(scaf_smi)
    r_scaf_dict[key_].add(scaf_smi)

avg_tmp = []
for k, v in r_recovered_dict.items():
    avg_tmp.append(len(v) / len(r_scaf_dict[k]))
recovery = sum(avg_tmp) / len(avg_tmp) * 100
print(f'Recovery: {recovery:.3f}%')
summary['recovery'] = recovery

# ---------------------------- Saving -------------------------------- #

# out_path = gen_smi_file[:-3] + 'csv'
# table = pd.DataFrame(data)
# table.to_csv(out_path, index=False)

# summary_path = gen_smi_file[:-4] + '_summary.csv'
# summary_table = pd.DataFrame([summary])
# summary_table.to_csv(summary_path, index=False)
