
import pandas as pd
from Bio import SeqIO
import torch
from io import StringIO
import numpy as np
from rdkit import Chem
import esm
from torch.utils.data import DataLoader
import pickle
from model import custom_collate,DrugProteinDataset,protein_graph_to_data,drug_graph_to_data
from rdkit.Chem import SanitizeMol, SanitizeFlags



# Constants
l_sub=10
N=193
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_virus(fasta_content) :
    lines = fasta_content.strip().splitlines()
    return ''.join(line.strip() for line in lines if not line.startswith(">"))

def read_virus_seqs(fasta_content):
    handle = StringIO(fasta_content)
    
    records = list(SeqIO.parse(handle, "fasta"))
    df = pd.DataFrame.from_records([
        {
            "ID": "|".join(record.description.split("|")[0:]),
            "Sequence": str(record.seq),
        }
        for record in records
    ])
    df["Virus_ID"] = df["ID"].apply(lambda x: "".join(x.split("|")[1:]) if "|" in x else "")
    df["Seq_ID"] = df["ID"].apply(lambda x: x.split("|")[0] if "|" in x else "")
   
    return df 

def ASW(sequence):
    l = len(sequence)
    if N> 1:
        l_stride = (l - l_sub) // (N - 1)
    else:
        l_stride = 1  

    subsequences = []

    for i in range(0, min(N * l_stride, l - l_sub + 1), l_stride):
        subsequences.append(sequence[i:i + l_sub])

    return subsequences

def tokenize(datas,ids,seq_ids,labels,ft_model):
    datas = [ASW(sequence) for sequence in datas.tolist()]
    labels= np.repeat(labels, N).tolist()
    ids=np.repeat(ids, N).tolist()
    seq_ids=np.repeat(seq_ids, N).tolist()

    embeddings = np.array([
        ft_model.wv[k]  # FastText will handle unknown k-mers
        for kmer in datas
        for k in kmer
    ])

    embeddings=torch.tensor(embeddings).to(device)
    ids=torch.tensor(ids).to(device)
    seq_ids=torch.tensor(seq_ids).to(device)
    labels=torch.tensor(labels).to(device)
    return embeddings,ids,seq_ids,labels

def test_one_virus(model,ft_model,datas,ids,seq_ids,labels=None):
    model.eval()
    if not labels:
        labels=np.zeros(len(ids))
    embeddings, ids,seq_ids, labels=tokenize(datas,ids,seq_ids,labels,ft_model)
    with torch.no_grad():
        Y_prob, Y_hat, A, A_2 =model(embeddings,ids,seq_ids)         
    return Y_prob, Y_hat, A, A_2 

def one_hot_encode(value, valid_values):
    if value not in valid_values:
        value = valid_values[-1]
    return [value == item for item in valid_values]


def get_atom_features(atom):
    atom_symbols = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
        'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
        'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
        'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'X'
    ]
    degrees = list(range(11))
    hydrogen_counts = list(range(11))
    valences = list(range(11))

    features = (
        one_hot_encode(atom.GetSymbol(), atom_symbols) +
        one_hot_encode(atom.GetDegree(), degrees) +
        one_hot_encode(atom.GetTotalNumHs(), hydrogen_counts) +
        one_hot_encode(atom.GetImplicitValence(), valences) +
        [atom.GetIsAromatic()]
    )

    return np.array(features)
    
def smile_graph(smile):
    nodes=[]
    edges=[]
    edges_type=[]
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        nodes.append(get_atom_features(atom))
    
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()

        # Since molecular graphs are undirected, add both directions
        edges.append([start, end])
        edges.append([end, start])

        edges_type.append(bond_type)
        edges_type.append(bond_type)
        

    return mol_size,nodes,edges,edges_type


def split_sequence(seq, window_size=1000, stride=500):
    windows = []
    for start in range(0, len(seq), stride):
        end = min(start + window_size, len(seq))
        if end - start < 2:  # skip too-short fragments
            break
        windows.append((start, seq[start:end]))
        if end == len(seq):
            break
    return windows

def esm_model_func(model,alphabet,seq):
   
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", seq)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[16], return_contacts=True)
    
    contact_map = results["contacts"]  # Shape: [1, L, L]
    
    return contact_map

def protein_graph(model, alphabet, seq, threshold=0.5, window_size=1000, stride=500):
    aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    L = len(seq)
    
    # Build node features (one-hot encoding for the full sequence)
    node_features = torch.eye(20)[[aa_dict.get(aa, 0) for aa in seq]]  # [L, 20]

    # Containers for merged edges
    edge_index = []
    edge_attr = []

    windows = split_sequence(seq, window_size, stride)

    for start_idx, subseq in windows:
        contact_map = esm_model_func(model, alphabet, subseq)[0]  # shape: [L_window, L_window]
        L_win = len(subseq)
        for i in range(L_win):
            for j in range(L_win):
                prob = contact_map[i, j].item()
                if prob > threshold:
                    global_i = start_idx + i
                    global_j = start_idx + j
                    if global_i < L and global_j < L:
                        edge_index.append([global_i, global_j])
                        edge_attr.append(prob)

    return node_features, edge_index, edge_attr


def test_antivirus(model,esm_model,esm_alphabet,virus,smiles):
    virus_graph=protein_graph(esm_model,esm_alphabet,virus)
    drug_graph=smile_graph(smiles)
    
    virus_graph = protein_graph_to_data(virus_graph).to(device)
    drug_graph = drug_graph_to_data(drug_graph).to(device)

    model.eval()
    with torch.no_grad():
        output = model(virus_graph, drug_graph)
    return output

def test_top_antivirus(model,esm_model,esm_alphabet,virus):
    with open(r"data/drug_graphs.pkl", "rb") as f:
        drug_graphs = pickle.load(f)
    drugs=pd.read_csv(r"data/drugs.csv")
    drugs = drugs['SMILES'].tolist()

    virus_graph = protein_graph(esm_model, esm_alphabet, virus)
    virus_graph = [virus_graph] * len(drug_graphs)
    dataset= DrugProteinDataset(virus_graph, drug_graphs)
    loader=DataLoader(dataset, batch_size=64, collate_fn=custom_collate)
    model.eval()
    outputs=[]
    for protein_graphs, drug_graphs in loader:
            protein_graphs = protein_graphs.to(device)
            drug_graphs = drug_graphs.to(device)
            with torch.no_grad():
                output = model(protein_graphs, drug_graphs)
            outputs.extend(output.cpu().detach().numpy())
    outputs = np.array(outputs)
    top5_indices = outputs.argsort()[::-1][:5]  # Descending order
    top5_smiles = [drugs[i] for i in top5_indices]

    return top5_smiles

