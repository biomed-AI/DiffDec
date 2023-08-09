import torch
from rdkit import Chem

split_by_name = torch.load('split_by_name.pt')
path_prefix = 'data/crossdocked_pocket10/'
f = open('train_smi.smi', 'w')
for i in range(len(split_by_name['train'])):
    ligand_filename = path_prefix + split_by_name['train'][i][1]
    mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    f.write(smi + '\n')
f.close()
f = open('test_smi.smi', 'w')
for i in range(len(split_by_name['test'])):
    ligand_filename = path_prefix + split_by_name['test'][i][1]
    mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    f.write(smi + '\n')
f.close()
