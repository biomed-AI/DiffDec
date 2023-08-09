import os
import torch
import numpy as np
from rdkit import Chem, Geometry
from Bio.PDB import PDBParser
import itertools
import pandas as pd
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_pocket(mol, pdb_path):
    struct = PDBParser().get_structure('', pdb_path)
    residue_ids = []
    atom_coords = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())
            residue_ids.append(resid)

    residue_ids = np.array(residue_ids)
    atom_coords = np.array(atom_coords)
    mol_atom_coords = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(atom_coords[:, None, :] - mol_atom_coords[None, :, :], axis=-1)
    contact_residues = np.unique(residue_ids[np.where(distances.min(1) <= 6)[0]])

    pocket_coords_full = []
    pocket_types_full = []

    pocket_coords_bb = []
    pocket_types_bb = []

    for residue in struct.get_residues():
        resid = residue.get_id()[1]
        if resid not in contact_residues:
            continue

        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            atom_type = atom.element.upper()
            atom_coord = atom.get_coord()

            pocket_coords_full.append(atom_coord.tolist())
            pocket_types_full.append(atom_type)

            if atom_name in {'N', 'CA', 'C', 'O'}:
                pocket_coords_bb.append(atom_coord.tolist())
                pocket_types_bb.append(atom_type)

    return {
        'full_coord': pocket_coords_full,
        'full_types': pocket_types_full,
        'bb_coord': pocket_coords_bb,
        'bb_types': pocket_types_bb,
    }

def get_exits(mol):
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

def set_anchor_flags(mol, anchor_idx):
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            if atom.HasProp('_Anchor'):
                anchor_num = int(atom.GetProp('_Anchor')) + 1
                atom.SetProp('_Anchor', str(anchor_num))
            else:
                atom.SetProp('_Anchor', '1')
        else:
            if not atom.HasProp('_Anchor'):
                atom.SetProp('_Anchor', '0')

def update_scaffold(scaffold):
    exits = get_exits(scaffold)

    # Sort exit atoms by id for further correct deletion
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)

    # Remove exit bonds
    for exit in exits:
        exit_idx = exit.GetIdx()
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        anchor_idx = source_idx if target_idx == exit_idx else target_idx
        set_anchor_flags(scaffold, anchor_idx)

    escaffold = Chem.EditableMol(scaffold)
    for exit in exits:
        exit_idx = exit.GetIdx()
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        escaffold.RemoveBond(source_idx, target_idx)

    # Remove exit atoms
    for exit in exits:
        escaffold.RemoveAtom(exit.GetIdx())

    return escaffold.GetMol()

def update_rgroups(scaffold, mol):
    rgroups = Chem.DeleteSubstructs(mol, scaffold)
    rgroups_ = Chem.GetMolFrags(rgroups, asMols=True, sanitizeFrags=False)
    return rgroups

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)

    if len(matches) < 1:
        raise Exception('Could not find scaffold or rgroup matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        scaf_coords = mol_coords[np.array(match)]
        scaf_conformer = create_conformer(scaf_coords)
        match2conf[match] = scaf_conformer

    return match2conf

def transfer_conformers_rgroups(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)
    if len(matches) < 1:
        raise Exception('Could not find scaffold or rgroup matches')
    atom_num = mol.GetNumAtoms()
    matches = matches[0]
    match_rgroup = []
    for i in range(atom_num):
        if i not in matches:
            match_rgroup.append(i)
    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        scaf_coords = mol_coords[np.array(match)]
        scaf_conformer = create_conformer(scaf_coords)
        match2conf[match] = scaf_conformer

    return match2conf

def find_non_intersecting_matches(matches1, matches2):
    triplets = list(itertools.product(matches1, matches2))
    non_intersecting_matches = set()
    for m1, m2 in triplets:
        m1m2 = set(m1) & set(m2)
        if len(m1m2) == 0:
            non_intersecting_matches.add((m1, m2))
    return list(non_intersecting_matches)

def find_matches_with_rgroup_in_the_middle(non_intersecting_matches, mol):
    matches_with_rgroup_in_the_middle = []
    for m1, lm in non_intersecting_matches:
        neighbors = set()
        for atom_idx in lm:
            atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in atom_neighbors:
                neighbors.add(neighbor.GetIdx())

        conn1 = set(m1) & neighbors
        if len(conn1) == 1:
            matches_with_rgroup_in_the_middle.append((m1, lm))

    return matches_with_rgroup_in_the_middle

def find_correct_matches(matches_scaf, matches_rgroup, mol):
    non_intersecting_matches = find_non_intersecting_matches(matches_scaf, matches_rgroup)
    if len(non_intersecting_matches) == 1:
        return non_intersecting_matches

    return find_matches_with_rgroup_in_the_middle(non_intersecting_matches, mol)

def prepare_scaffold_and_rgroup_list(scaf_smi, rgroup_smi_list, mol):
    scaf = Chem.MolFromSmiles(scaf_smi)

    newscaf = update_scaffold(scaf)
    newrgroup = update_rgroups(newscaf, mol)

    match2conf_scaf = transfer_conformers(newscaf, mol)
    match2conf_rgroup = transfer_conformers(newrgroup, mol)

    correct_matches = find_correct_matches(
        match2conf_scaf.keys(),
        match2conf_rgroup.keys(),
        mol,
    )

    if len(correct_matches) > 2:
        raise Exception('Found more than two scaffold matches')

    conf_scaf = match2conf_scaf[correct_matches[0][0]]
    conf_rgroup = match2conf_rgroup[correct_matches[0][1]]
    newscaf.AddConformer(conf_scaf)
    newrgroup.AddConformer(conf_rgroup)

    return newscaf, newrgroup

def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        anchor_num = int(atom.GetProp('_Anchor'))
        if anchor_num != 0:
            for _ in range(anchor_num):
                anchors_idx.append(atom.GetIdx())

    return anchors_idx


def process_sdf(scaf_dataset):
    molecules = []
    scaffolds = []
    rgroups = []
    pockets = []
    out_table = []

    uuid = 0

    scaf_dataset = torch.load(scaf_dataset)

    for i in range(len(scaf_dataset['scaf_smi'])):
        ligand_filename = os.path.join('/home/xiejunjie/project/flag/data/crossdocked_pocket10',
                            scaf_dataset['ligand_filename'][i])
        protein_filename = os.path.join('/home/xiejunjie/project/flag/data/crossdocked_pocket10',
                            scaf_dataset['protein_filename'][i])
        scaf_smi = scaf_dataset['scaf_smi'][i]
        rgroup_smi_list = scaf_dataset['rgroup_smi'][i]

        try:
            mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
            Chem.SanitizeMol(mol)
        except:
            continue
        if mol is None:
            continue

        mol_name = mol.GetProp('_Name')
        mol_smi = Chem.MolToSmiles(mol)
        pocket = get_pocket(mol, protein_filename)
        if len(pocket['full_coord']) == 0:
            continue

        try:
            scaffold, rgroup = prepare_scaffold_and_rgroup_list(scaf_smi, rgroup_smi_list, mol)
        except Exception as e:
            print(f'{mol_smi} | {scaf_smi} | {rgroup_smi_list[0]} : {e}')
            continue
    
        anchors_idx = get_anchors_idx(scaffold)

        anchors_str = str(anchors_idx[0])
        for j in range(1, len(anchors_idx)):
            anchors_str += '|'
            anchors_str += str(anchors_idx[j])

        molecules.append(mol)
        scaffolds.append(scaffold)
        rgroups.append(rgroup)
        pockets.append(pocket)
        out_table.append({
            'uuid': uuid,
            'molecule_name': mol_name,
            'molecule': mol_smi,
            'scaffold': Chem.MolToSmiles(scaffold),
            'rgroups': Chem.MolToSmiles(rgroup),
            'anchor': anchors_str,
            'pocket_full_size': len(pocket['full_types']),
            'pocket_bb_size': len(pocket['bb_types']),
            'molecule_size': mol.GetNumAtoms(),
            'scaffold_size': scaffold.GetNumAtoms(),
            'rgroup_size': rgroup.GetNumAtoms(),
            'protein_filename': protein_filename,
        })
        uuid += 1

    return molecules, scaffolds, rgroups, pockets, pd.DataFrame(out_table)

def prepare(sliced_file, mode):
    decomp_dict = dict()
    with open(sliced_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(sep='\t')
            rgroups_list = line[1].strip().split(sep='|')
            if line[2] not in decomp_dict:
                decomp_dict[line[2]] = [(line[0], rgroups_list)]
            else:
                decomp_dict[line[2]].append((line[0], rgroups_list))

    scaf_dict = {
        'ligand_filename': [],
        'protein_filename': [],
        'scaf_smi': [],
        'rgroup_smi': [],
    }
    split_by_name = torch.load('split_by_name.pt')
    path_prefix = 'data/crossdocked_pocket10/'
    for i in range(len(split_by_name[mode])):
        ligand_filename = path_prefix + split_by_name[mode][i][1]
        mol = next(iter(Chem.SDMolSupplier(ligand_filename, removeHs=True)))
        smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        if smi not in decomp_dict:
            continue
        else:
            decomp_list = decomp_dict[smi]
            for j in range(len(decomp_list)):
                scaf = decomp_list[j][0]
                rgroup_list = decomp_list[j][1]
                scaf_dict['ligand_filename'].append(split_by_name[mode][i][1])
                scaf_dict['protein_filename'].append(split_by_name[mode][i][0])
                scaf_dict['scaf_smi'].append(scaf)
                scaf_dict['rgroup_smi'].append(rgroup_list)
    return scaf_dict

def main(scaf_dataset, mode):
    out_mol_path = './crossdock_' + mode +'_mol.sdf'
    out_scaf_path = './crossdock_' + mode +  '_scaf.sdf'
    out_rgroup_path = './crossdock_' + mode +  '_rgroup.sdf'
    out_pockets_path = './crossdock_' + mode +  '_pockets.pkl'
    out_table_path = './crossdock_' + mode +  '_table.csv'
    molecules, scaffolds, rgroups, pockets, out_table = process_sdf(scaf_dataset)
    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        for i, mol in enumerate(molecules):
            writer.write(mol)
    with Chem.SDWriter(open(out_scaf_path, 'w')) as writer:
        writer.SetKekulize(False)
        for i, scaf in enumerate(scaffolds):
            writer.write(scaf)
    with Chem.SDWriter(open(out_rgroup_path, 'w')) as writer:
        writer.SetKekulize(False)
        for i, rgroup in enumerate(rgroups):
            writer.write(rgroup)
    with open(out_pockets_path, 'wb') as f:
        pickle.dump(pockets, f)

    out_table = out_table.reset_index(drop=True)
    out_table.to_csv(out_table_path, index=False)

if __name__ == '__main__':
    train_sliced_file = 'multi_train.smi'
    test_sliced_file = 'multi_test.smi'
    scaf_train = prepare(train_sliced_file, 'train')
    scaf_test = prepare(test_sliced_file, 'test')
    main(scaf_train, 'train')
    main(scaf_test, 'test')