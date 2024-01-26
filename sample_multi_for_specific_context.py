import os
import torch
import numpy as np
from rdkit import Chem, Geometry
from Bio.PDB import PDBParser
import itertools
import pandas as pd
import pickle
import argparse

from src.model_multi import DDPM
from src.visualizer import save_xyz_file_fa
from src.datasets import collate_mr
from tqdm import tqdm

import subprocess

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

        return scaf_conformer

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

def prepare_scaffold_and_rgroup(scaf_smi, mol):
    scaf = Chem.MolFromSmiles(scaf_smi)
    newscaf = update_scaffold(scaf)
    scaf_conformer = transfer_conformers(newscaf, mol)
    newscaf.AddConformer(scaf_conformer)

    return newscaf

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

    for i in range(len(scaf_dataset['scaf_smi'])):
        ligand_filename = scaf_dataset['ligand_filename'][i]
        protein_filename = scaf_dataset['protein_filename'][i]
        scaf_smi = scaf_dataset['scaf_smi'][i]
        rgroup_smi = scaf_dataset['rgroup_smi'][i]

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
            scaffold = prepare_scaffold_and_rgroup(scaf_smi, mol)
        except Exception as e:
            print(f'{mol_smi} | {scaf_smi} | {rgroup_smi} : {e}')
            continue
        
        anchors_idx = get_anchors_idx(scaffold)
        anchors_str = str(anchors_idx[0])
        for j in range(1, len(anchors_idx)):
            anchors_str += '|'
            anchors_str += str(anchors_idx[j])

        rgroups_str = 'C'
        for j in range(1, len(anchors_idx)):
            rgroups_str += '.C'

        molecules.append(mol)
        scaffolds.append(scaffold)
        rgroups.append(Chem.MolFromSmiles(rgroups_str))
        pockets.append(pocket)
        out_table.append({
            'uuid': uuid,
            'molecule_name': mol_name,
            'molecule': mol_smi,
            'scaffold': Chem.MolToSmiles(scaffold),
            'rgroups': None,
            'anchor': anchors_str,
            'pocket_full_size': len(pocket['full_types']),
            'pocket_bb_size': len(pocket['bb_types']),
            'molecule_size': mol.GetNumAtoms(),
            'scaffold_size': scaffold.GetNumAtoms(),
            'rgroup_size': 0,
            'protein_filename': protein_filename,
        })
        uuid += 1

    return molecules, scaffolds, rgroups, pockets, pd.DataFrame(out_table)

def prepare(scaffold_smiles_file, protein_file, scaffold_file, task_name, data_dir, mode = 'test'):
    f = open(scaffold_smiles_file, 'r')
    scaffold_smiles = f.readlines()[0].strip()
    scaf_dict = {
        'ligand_filename': [],
        'protein_filename': [],
        'scaf_smi': [],
        'rgroup_smi': [],
    }
    scaf_dict['ligand_filename'].append(scaffold_file)
    scaf_dict['protein_filename'].append(protein_file)
    scaf_dict['scaf_smi'].append(scaffold_smiles)
    scaf_dict['rgroup_smi'].append('')

    out_mol_path = os.path.join(data_dir, task_name + '_' + mode +'_mol.sdf')
    out_scaf_path = os.path.join(data_dir, task_name + '_' + mode +  '_scaf.sdf')
    out_rgroup_path = os.path.join(data_dir, task_name + '_' + mode +  '_rgroup.sdf')
    out_pockets_path = os.path.join(data_dir, task_name + '_' + mode +  '_pockets.pkl')
    out_table_path = os.path.join(data_dir, task_name + '_' + mode +  '_table.csv')

    molecules, scaffolds, rgroups, pockets, out_table = process_sdf(scaf_dict)
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

def check_if_generated(_output_dir, _uuids, n_samples):
    generated = True
    starting_points = []
    for _uuid in _uuids:
        uuid_dir = os.path.join(_output_dir, _uuid)
        numbers = []
        for fname in os.listdir(uuid_dir):
            try:
                num = int(fname.split('_')[0])
                numbers.append(num)
            except:
                continue
        if len(numbers) == 0 or max(numbers) != n_samples - 1:
            generated = False
            if len(numbers) == 0:
                starting_points.append(0)
            else:
                starting_points.append(max(numbers) - 1)

    if len(starting_points) > 0:
        starting = min(starting_points)
    else:
        starting = None

    return generated, starting

def sample(checkpoint, samples_dir, data_dir, n_samples, task_name, device):
    experiment_name = checkpoint.split('/')[-1].replace('.ckpt', '')
    output_dir = os.path.join(samples_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    collate_fn = collate_mr
    sample_fn = None

    # Loading model form checkpoint (all hparams will be automatically set)
    model = DDPM.load_from_checkpoint(checkpoint, map_location=device)

    # Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
    model.val_data_prefix = task_name + '_test_full'

    # In case <Anonymous> will run my model or vice versa
    if data_dir is not None:
        model.data_path = data_dir

    # Setting up the model
    model = model.eval().to(device)
    model.setup(stage='val')

    model.batch_size = 1
    # Getting the dataloader
    dataloader = model.val_dataloader(collate_fn=collate_fn)
    print(f'Dataloader contains {len(dataloader)} batches')

    center_of_mass_list = []

    for batch_idx, data in enumerate(dataloader):
        uuids = []
        true_names = []
        scaf_names = []
        pock_names = []
        for uuid in data['uuid']:
            uuid = str(uuid)
            uuids.append(uuid)
            true_names.append(f'{uuid}/true')
            scaf_names.append(f'{uuid}/scaf')
            pock_names.append(f'{uuid}/pock')
            os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

        generated, starting_point = check_if_generated(output_dir, uuids, n_samples)
        if generated:
            print(f'Already generated batch={batch_idx}, max_uuid={max(uuids)}')
            continue
        if starting_point > 0:
            print(f'Generating {n_samples - starting_point} for batch={batch_idx}')

        h, x, node_mask, scaf_mask = data['one_hot'], data['positions'], data['atom_mask'], data['scaffold_mask']

        node_mask = data['atom_mask'] - data['pocket_mask']
        scaf_mask = data['scaffold_only_mask']
        pock_mask = data['pocket_mask']
        save_xyz_file_fa(output_dir, h, x, pock_mask, pock_names)

        # Saving ground-truth molecules
        save_xyz_file_fa(output_dir, h, x, node_mask, true_names)

        # Saving scaffold
        save_xyz_file_fa(output_dir, h, x, scaf_mask, scaf_names)

        # Sampling and saving generated molecules
        for i in tqdm(range(starting_point, n_samples), desc=str(batch_idx)):
            chain, node_mask, mean = model.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
            x = chain[-1][:, :, :model.n_dims]
            h = chain[-1][:, :, model.n_dims:]

            x += mean
            x_rgroup_tmp = x * data['rgroup_mask_batch_new']
            x_scaf_ori_tmp = data['positions'] * data['scaffold_mask']
            cnt = 0 
            for k in range(data['batch_new_len_tensor'].shape[0]):
                for j in range(data['batch_new_len_tensor'][k]):
                    x_scaf_ori_tmp[k] += x_rgroup_tmp[cnt]
                    cnt += 1

            h_rgroup_tmp = h * data['rgroup_mask_batch_new']
            h_scaf_ori_tmp = data['one_hot'] * data['scaffold_mask']
            cnt = 0 
            for k in range(data['batch_new_len_tensor'].shape[0]):
                for j in range(data['batch_new_len_tensor'][k]):
                    h_scaf_ori_tmp[k] += h_rgroup_tmp[cnt]
                    cnt += 1

            x = x_scaf_ori_tmp
            h = h_scaf_ori_tmp

            node_mask = data['atom_mask'] - data['pocket_mask']

            pred_names = [f'{uuid}/{i}' for uuid in uuids]
            
            save_xyz_file_fa(output_dir, h, x, node_mask, pred_names)
            for j in range(len(pred_names)):
                out_xyz = f'{output_dir}/{pred_names[j]}_.xyz'
                out_sdf = f'{output_dir}/{pred_names[j]}_.sdf'
                subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaffold_smiles_file', action='store', type=str, required=True)
    parser.add_argument('--protein_file', action='store', type=str, required=True)
    parser.add_argument('--scaffold_file', action='store', type=str, required=True)
    parser.add_argument('--task_name', action='store', type=str, required=True)
    parser.add_argument('--data_dir', action='store', type=str, required=True)

    parser.add_argument('--checkpoint', action='store', type=str, required=True)
    parser.add_argument('--samples_dir', action='store', type=str, required=True)
    parser.add_argument('--n_samples', action='store', type=int, required=True)
    parser.add_argument('--device', action='store', type=str, required=True)
    args = parser.parse_args()

    prepare(args.scaffold_smiles_file, args.protein_file, args.scaffold_file, args.task_name, args.data_dir)
    
    sample(args.checkpoint, args.samples_dir, args.data_dir, args.n_samples, args.task_name, args.device)