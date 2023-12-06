
from operator import index
import torch
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
import re


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):

    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
            one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)
    #print(torch.from_numpy(results))
    return torch.from_numpy(results)

def generate_drug_data(mol_graph, atom_symbols,smiles,id):
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    #print(edge_list.shape)
    #print(edge_list)
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats]*2, dim=0) if len(edge_feats) else edge_feats
    contribs = rdMolDescriptors._CalcCrippenContribs(mol_graph)
    chemfeat = torch.tensor(contribs[:3])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)
    fp = torch.tensor(fp)
    #print(smiles)
    #smword = splitSmi(smiles)
    #smvec = fetchIndices(smword , word , 94)
    #smvec = smvec.squeeze()
    #smvec = torch.tensor(smvec)
    if chemfeat.shape[0]==1:
        chemfeat = torch.reshape(chemfeat, [2])
        a = torch.tensor([0,0,0,0])
        chemfeat = torch.cat((chemfeat,a),dim = 0)
    elif chemfeat.shape[0]==2:
        chemfeat = torch.reshape(chemfeat, [4])
        a = torch.tensor([0,0])
        chemfeat = torch.cat((chemfeat,a),dim = 0)
    else:
        chemfeat = torch.reshape(chemfeat, [6])
    #print(type(contribs[:3]))
    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)
    #print(features)
    #print(type(features))
    line_graph_edge_index = torch.LongTensor([])

    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    '''print(edge_list[:, 1].unsqueeze(1))
    print(edge_list[:, 0].unsqueeze(0))
    print(edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0))
    print(edge_list[:, 0].unsqueeze(1))
    print(edge_list[:, 1].unsqueeze(0))
    print(edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
    print(conn)
    print(line_graph_edge_index)'''
    new_edge_index = edge_list.T
    #print(features.shape)
    #print(new_edge_index.reshape(2,-1).shape)
    #print(line_graph_edge_index.shape)
    #print(edge_feats.shape)
    return features, new_edge_index.reshape(2,-1), edge_feats, line_graph_edge_index, chemfeat , smiles , fp

def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()

def load_drug_mol_data():

    data = pd.read_csv(r'D:\code\DMI\data\drug_smiles1.csv')
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}

    for id, smiles in zip(data['id'], data['smiles']):
        drug_smile_dict[id] = smiles
    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id,smiles ,mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

    #print(len(drug_id_mol_tup))

    symbols = list(set(symbols))
    drug_data = {id: generate_drug_data(mol, symbols,smiles,id) for id, smiles,mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    #save_data(drug_data, 'drug_data.pkl', args)
    #print(drug_data[994][0],drug_data[994][1].reshape(2,-1))
    return drug_data

#load_drug_mol_data()