from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
import subprocess
from rdkit.Chem.rdMolAlign import CalcRMS
import numpy as np
from easydict import EasyDict
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit import Geometry
from rdkit.Chem.QED import qed
from rdkit.Chem import rdMolDescriptors
from rdkit.six.moves import cPickle
from rdkit.six import iteritems
import math
import os.path as op
import os
import string
import random
from rdkit import Chem
from copy import deepcopy
import csv
import torch
from Bio.PDB import PDBParser
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore")
from multiprocessing.dummy import Pool as ThreadPool
import time

_fscores = None

def get_logp(mol):
    return Crippen.MolLogP(mol)

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

def readFragmentScores(name='fpscores'):
  import gzip
  global _fscores
  # generate the full path filename:
  if name == "fpscores":
    name = op.join(op.dirname(__file__), name)
  _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
  outDict = {}
  for i in _fscores:
    for j in range(1, len(i)):
      outDict[i[j]] = float(i[0])
  _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead, nSpiro

def calculateScore(m):
  if _fscores is None:
    readFragmentScores()

  fp = rdMolDescriptors.GetMorganFingerprint(m,
                                             2)  #<- 2 is the *radius* of the circular fingerprint
  fps = fp.GetNonzeroElements()
  score1 = 0.
  nf = 0
  for bitId, v in iteritems(fps):
    nf += v
    sfp = bitId
    score1 += _fscores.get(sfp, -4) * v
  score1 /= nf

  # features score
  nAtoms = m.GetNumAtoms()
  nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
  ri = m.GetRingInfo()
  nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
  nMacrocycles = 0
  for x in ri.AtomRings():
    if len(x) > 8:
      nMacrocycles += 1

  sizePenalty = nAtoms**1.005 - nAtoms
  stereoPenalty = math.log10(nChiralCenters + 1)
  spiroPenalty = math.log10(nSpiro + 1)
  bridgePenalty = math.log10(nBridgeheads + 1)
  macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
  if nMacrocycles > 0:
    macrocyclePenalty = math.log10(2)

  score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
  score3 = 0.
  if nAtoms > len(fps):
    score3 = math.log(float(nAtoms) / len(fps)) * .5

  sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
  min = -4.0
  max = 2.5
  sascore = 11. - (sascore - min + 1) / (max - min) * 9.
  # smooth the 10-end
  if sascore > 8.:
    sascore = 8. + math.log(sascore + 1. - 9.)
  if sascore > 10.:
    sascore = 10.0
  elif sascore < 1.:
    sascore = 1.0

  return sascore

def compute_sa_score(rdmol):
    rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
    sa = calculateScore(rdmol)
    sa_norm = round((10-sa)/9,2)
    return sa_norm

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length)) 

def parse_qvina_outputs(docked_sdf_path, ref_mol):

    suppl = Chem.SDMolSupplier(docked_sdf_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except Exception as e:
            rmsd = np.nan
            # print('*parse_qvina*', e)
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            'rmsd_ref': rmsd
        }))

    return results

class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()
    
    def get_results(self):
        raise NotImplementedError()


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_data(cls, ligand_mol, protein_path, ligand_path):
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        struct = PDBParser().get_structure('', protein_path)
        # ligand_rdmol = Chem.MolFromSmiles(ligand_smi)
        # AllChem.EmbedMolecule(ligand_rdmol)

        return cls(pdb_block, ligand_mol, ligand_path, struct)

    def __init__(self, pdb_block, ligand_rdmol, ligand_path, struct, conda_env='adt', tmp_dir='./tmp', use_uff=True, center=None):
        super().__init__(pdb_block, ligand_rdmol)

        residue_ids = []
        atom_coords = []

        for residue in struct.get_residues():
            resid = residue.get_id()[1]
            for atom in residue.get_atoms():
                atom_coords.append(atom.get_coord())
                residue_ids.append(resid)

        residue_ids = np.array(residue_ids)
        atom_coords = np.array(atom_coords)
        center_pro = (atom_coords.max(0) + atom_coords.min(0)) / 2
        # print('1:', center_pro)

        # mol_ref = next(iter(Chem.SDMolSupplier(ligand_path)))
        # pos_ref = mol_ref.GetConformer(0).GetPositions()
        # center_ref = (pos_ref.max(0) + pos_ref.min(0)) / 2
        # print('2:', center_ref)

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        AllChem.EmbedMolecule(ligand_rdmol)
        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        # ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if use_uff:
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(ligand_rdmol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol
        self.noH_rdmol = Chem.RemoveHs(ligand_rdmol)

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center
        
        self.center = center_pro

        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_path = None
    
    def run(self, exhaustiveness=16):
        commands = """
eval "$(conda shell.bash hook)"
conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
prepare_receptor4.py -r {receptor_id}.pdb -o {receptor_id}.pdbqt
# Prepare ligand
obabel {ligand_id}.sdf -O{ligand_id}.pdbqt
qvina2 \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x 30 --size_y 30 --size_z 30 \
    --exhaustiveness {exhaust} \
    --cpu 1 \
    --seed 1 
obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h
        """.format(
            receptor_id = self.receptor_id,
            ligand_id = self.ligand_id,
            env = self.conda_env, 
            tmp = self.tmp_dir, 
            exhaust = exhaustiveness,
            center_x = self.center[0],
            center_y = self.center[1],
            center_z = self.center[2],
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)

        self.proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))
        self.proc.stdin.close()

    def run_sync(self):
        self.run()
        while self.get_results() is None:
            pass
        results = self.get_results()
        return results

    def get_results(self):
        if self.proc is None:   # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                try:
                    self.results = parse_qvina_outputs(self.docked_sdf_path, self.noH_rdmol)
                except Exception as e:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)
                    print('*get_results*', e)
                    return []
            return self.results

def cal_vina_docking(
    test_csv_path = '',
    results_path = ''
):
    if os.path.exists(results_path):
        return
    pred_list = []
    protein_filename_list = []
    with open(test_csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            pred_list.append(row[2])
            protein_filename_list.append(row[4])

    result = []

    for i in range(len(pred_list)):
        pred_smi = pred_list[i]
        pred_mol = Chem.MolFromSmiles(pred_smi, sanitize=True)
        if pred_mol is None:
            pred_mol = Chem.MolFromSmiles(pred_smi, sanitize=False)
        ligand_filename = protein_filename_list[i][:-13] + '.sdf'
        if pred_mol is None or pred_smi == '':
            print(i, 'None')
            result.append({
                'mol': pred_mol,
                'vina': None,
                'i': i
            })
            return 
        try:
            vina_task = QVinaDockingTask.from_data(pred_mol, protein_filename_list[i], ligand_filename)
            vina = vina_task.run_sync()
            result.append({
                'mol': pred_mol,
                'vina': vina,
                'i': i
            })
        except Exception as e:
            print(e)
            print('Failed %d' % i)
            result.append({
                'mol': pred_mol,
                'vina': None,
                'i': i
            })
        if i % 200 == 0:
            torch.save(result, results_path)

    torch.save(result, results_path)

def cal_average_metric(
    results_path = 'result.pt',
    test_csv_path = '',
):
    sample_num = 100
    protein_filename_list = []
    true_smi_list = []
    cnt = 0
    with open(test_csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if cnt % sample_num == 0:
                true_smi_list.append(row[1])
                protein_filename_list.append(row[4])
            cnt += 1
    vina_metric_dict = {}
    for i in range(len(true_smi_list)):
        vina_metric_dict[true_smi_list[i] + protein_filename_list[i]] = []

    results = torch.load(results_path)
    for i in range(len(results)):
        idx = results[i]['i']
        if (results[i]['vina'] != None) and (len(results[i]['vina']) != 0):
            vina_metric_dict[true_smi_list[int(idx / sample_num)] + protein_filename_list[int(idx / sample_num)]].append(results[i]['vina'][0]['affinity'])
    avg_tmp = []
    for k, v in vina_metric_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    print('vina score:', sum(avg_tmp) / len(avg_tmp))
    print('='*30)

def cal_test_metric(
    test_csv_path = 'crossdock_test.csv',
    results_path = 'result_testset.pt'
):
    sample_num = 100
    if os.path.exists(results_path):
        protein_filename_list = []
        true_smi_list = []
        cnt = 0
        with open(test_csv_path, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for row in csv_reader:
                if cnt % sample_num == 0:
                    true_smi_list.append(row[1])
                    protein_filename_list.append(row[4])
                cnt += 1
        vina_metric_dict = {}
        for i in range(len(true_smi_list)):
            vina_metric_dict[true_smi_list[i] + protein_filename_list[i]] = []

        results = torch.load(results_path)
        for i in range(len(results)):
            if (results[i]['vina'] != None) and (len(results[i]['vina']) != 0):
                vina_metric_dict[true_smi_list[i] + protein_filename_list[i]].append(results[i]['vina'][0]['affinity'])
        avg_tmp = []
        for k, v in vina_metric_dict.items():
            if len(v) == 0:
                continue
            avg_tmp.append(sum(v) / len(v))
        print('reference vina score:', sum(avg_tmp) / len(avg_tmp))
        print('='*30)
        return 
    
    protein_filename_list = []
    true_smi_list = []
    cnt = 0
    with open(test_csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if cnt % sample_num == 0:
                true_smi_list.append(row[1])
                protein_filename_list.append(row[4])
            cnt += 1
    
    vina_metric_dict = {}
    for i in range(len(true_smi_list)):
        vina_metric_dict[true_smi_list[i] + protein_filename_list[i]] = []

    result = []

    for i in range(len(true_smi_list)):
        smi = true_smi_list[i]
        pred_mol = Chem.MolFromSmiles(smi, sanitize=True)
        ligand_filename = protein_filename_list[i][:-13] + '.sdf'
        if pred_mol is None:
            pred_mol = Chem.MolFromSmiles(smi, sanitize=False)
        try:
            vina_task = QVinaDockingTask.from_data(pred_mol, protein_filename_list[i], ligand_filename)
            vina = vina_task.run_sync()
            result.append({
                'mol': pred_mol,
                'vina': vina,
                'i': i
            })
            vina_metric_dict[true_smi_list[i] + protein_filename_list[i]].append(vina[0]['affinity'])
        except Exception as e:
            print(e, smi)
            print('Failed %d' % i)
            result.append({
                'mol': pred_mol,
                'vina': None,
                'i': i
            })

    torch.save(result, results_path)
    
    avg_tmp = []
    for k, v in vina_metric_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    print('reference vina score:', sum(avg_tmp) / len(avg_tmp))
    print('='*30)

def cal_high_aff(
    results_testset_path = 'result_testset.pt',
    results_pred_path = 'result.pt',
    test_csv_path = ''
):
    sample_num = 100

    protein_filename_list = []
    true_smi_list = []
    cnt = 0
    with open(test_csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if cnt % sample_num == 0:
                true_smi_list.append(row[1])
                protein_filename_list.append(row[4])
            cnt += 1
    high_aff_cnt_dict = {}
    tot_cnt_dict = {}
    for i in range(len(true_smi_list)):
        high_aff_cnt_dict[true_smi_list[i] + protein_filename_list[i]] = 0
        tot_cnt_dict[true_smi_list[i] + protein_filename_list[i]] = 0

    results_testset = torch.load(results_testset_path)
    results_pred = torch.load(results_pred_path)

    for i in range(len(results_pred)):
        idx = int(results_pred[i]['i'] / sample_num)
        if results_pred[i]['vina'] != None:
            vina_score = results_pred[i]['vina'][0]['affinity']
            idx = int(results_pred[i]['i'] / sample_num)
            if results_testset[idx]['vina'] is None:
                continue
            vina_ref = results_testset[idx]['vina'][0]['affinity']
            if vina_score <= vina_ref:
                high_aff_cnt_dict[true_smi_list[idx] + protein_filename_list[idx]] += 1
        tot_cnt_dict[true_smi_list[idx] + protein_filename_list[idx]] += 1
    avg_tmp = []
    for k, v in high_aff_cnt_dict.items():
        if tot_cnt_dict[k] == 0:
            continue
        avg_tmp.append(v / tot_cnt_dict[k])
    print('high affinity:', sum(avg_tmp) / len(avg_tmp))

def formatted_main(test_csv_path, results_pred_path, results_test_path):
    cal_vina_docking(test_csv_path=test_csv_path, results_path=results_pred_path)
    cal_test_metric(test_csv_path=test_csv_path, results_path=results_test_path)
    cal_average_metric(results_path=results_pred_path, test_csv_path=test_csv_path)
    cal_high_aff(results_testset_path=results_test_path, results_pred_path=results_pred_path, test_csv_path=test_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv_path', action='store', type=str, required=False, default='formatted_fake_atom/crossdock_test.csv')
    parser.add_argument('--results_pred_path', action='store', type=str, required=False, default='formatted_fake_atom/result.pt')
    parser.add_argument('--results_test_path', action='store', type=str, required=False, default='formatted_fake_atom/result_testset.pt')
    args = parser.parse_args()
    test_csv_path = args.test_csv_path
    results_pred_path = args.results_pred_path
    results_test_path = args.results_test_path
    time_start = time.time()
    formatted_main(test_csv_path, results_pred_path, results_test_path)
    time_end = time.time()
    print('sample time:', time_end - time_start, 's')
    
    tmp_folder = os.path.join('tmp')
    try:
        shutil.rmtree(tmp_folder)
        print('deleted tmp files')
    except:
        print('Please delete by hand')
