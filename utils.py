
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
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64



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

def visualize_attention_2d_heatmaps(A, A_2, Seq_ids, super_ids, save_path=None, cmap='viridis', annot=False, 
                                normalize=True, threshold=0.1, title_prefix="Attention Heatmap", 
                                x_tick_step=10, figsize=(16, 8)):
    # Convert to numpy if not already
    A = np.array(A) if not isinstance(A, np.ndarray) else A
    A_2 = np.array(A_2) if not isinstance(A_2, np.ndarray) else A_2

    # Shape validation
    if A.ndim != 2 or A_2.ndim != 2:
        raise ValueError("A and A_2 must be 2D arrays")
    n_bags, n_instances = A.shape
    
    # Normalize attention weights
    if normalize:
        A = np.clip(A, 0, None)
        A_max = np.max(A) if np.max(A) > 0 else 1.0
        A = A / A_max if A_max > 0 else A

        A_2 = np.clip(A_2, 0, None)
        A_2_max = np.max(A_2) if np.max(A_2) > 0 else 1.0
        A_2 = A_2 / A_2_max if A_2_max > 0 else A_2

    # Create figure with two subplots and a bar plot
    img_io = BytesIO()
    plt.figure(figsize=figsize)
    A_2_ = np.array(A_2).reshape(-1)     # Ensure A_2 is 1D with shape (7,)
    combined_A = A * A_2_[:, np.newaxis]  # Multiply each row of A by the corresponding A_2 weight
    # Plot instance-level attention heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(A_2, cmap=cmap, annot=annot, fmt='.2f', cbar_kws={'label': 'Attention Weight'},
                xticklabels=Seq_ids)
    plt.title(f"{title_prefix} - Bag-Level (A_2)")
    plt.xlabel("Sequence IDs")
    plt.ylabel("Virus IDs")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    global_max_idx_2 = np.unravel_index(np.argmax(A_2), A_2.shape)
    plt.scatter(global_max_idx_2[1] + 0.5, global_max_idx_2[0] + 0.5, color='red', s=100, label='Max Sequence per Virus', zorder=5)
    plt.legend()


    # Plot bag-level attention heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(combined_A, cmap=cmap, annot=annot, fmt='.2f', cbar_kws={'label': 'Attention Weight'},
                yticklabels=Seq_ids)
    plt.title(f"{title_prefix} - Instance-Level (A)")
    plt.xlabel("Tokens (Instances)")
    plt.ylabel("Sequence IDs")
    plt.xticks(np.arange(0, n_instances, x_tick_step), rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Highlight global max attention and per-sequence max attention
    global_max_idx = np.unravel_index(np.argmax(A), A.shape)
    # plt.scatter(global_max_idx[1] + 0.5, global_max_idx[0] + 0.5, color='red', s=100, label='Global Max', zorder=5)
    for i in range(n_bags):
        seq_max_idx = np.argmax(combined_A[i, :])
        plt.scatter(seq_max_idx + 0.5, i + 0.5, color='yellow', s=50, label='Max Subsequence per Sequence' if i == 0 else "", zorder=5)
    plt.legend()
    # Highlight global max attention

    plt.tight_layout()
    plt.savefig(img_io, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    img_io.seek(0)
    encoded_img = base64.b64encode(img_io.read()).decode('utf-8')
    return encoded_img

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
    top5_smiles = [(drugs[i], float(outputs[i])) for i in top5_indices]

    return top5_smiles

