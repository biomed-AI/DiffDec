import torch

from rdkit import Chem

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8

# Atom idx for one-hot encoding '#'-fake atom
ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, '#': 9}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P', 9: '#'}

# Atomic numbers
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15, '#': 0}
CHARGES_LIST = [6, 8, 7, 9, 16, 17, 35, 53, 15, 0]

# One-hot atom types
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)

# Dataset keys
DATA_LIST_ATTRS = {
    'uuid', 'name', 'scaffold_smi', 'rgroup_smi', 'num_atoms', 'cat', 'rgroup_size', 'anchors_str', 'edge_index'
}
DATA_ATTRS_TO_PAD = {
    'positions', 'one_hot', 'charges', 'anchors', 'scaffold_mask', 'rgroup_mask', 'pocket_mask', 'scaffold_only_mask'
}
DATA_ATTRS_TO_ADD_LAST_DIM = {
    'charges', 'anchors', 'scaffold_mask', 'rgroup_mask', 'pocket_mask', 'scaffold_only_mask'
}

MARGINS_EDM = [10, 5, 2]