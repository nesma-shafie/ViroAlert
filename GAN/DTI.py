from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import esm
import numpy as np
def one_hot_encode(value, valid_values):
    encoded = []
    for item in valid_values:
        if value == item:
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded
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

def protein_graph_to_data(protein_graph):
    node_features,edge_index,edge_attr = protein_graph
    x = node_features
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def drug_graph_to_data(drug_graph):
    mol_size, nodes, edges, edges_type = drug_graph
    x = torch.tensor(nodes, dtype=torch.float)  # [num_nodes, node_features]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edges_type, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)  # bias removed

    @staticmethod
    def degree(index, num_nodes, dtype):
        out = torch.zeros(num_nodes, dtype=dtype, device=index.device)
        ones = torch.ones_like(index, dtype=dtype)
        out.scatter_add_(0, index, ones)
        return out

    @staticmethod
    def add_self_loops(edge_index, num_nodes):
        self_loops = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        self_loops = self_loops.unsqueeze(0).repeat(2, 1)
        return torch.cat([edge_index, self_loops], dim=1)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)

        # Add self-loops
        edge_index = self.add_self_loops(edge_index, num_nodes)
        self_loop_weight = torch.ones(num_nodes, device=x.device).unsqueeze(1)
        edge_weight = torch.cat([edge_weight, self_loop_weight], dim=0).squeeze()

        row, col = edge_index

        # Improved normalization stability
        deg = self.degree(col, num_nodes, dtype=x.dtype)
        deg = torch.clamp(deg, min=1e-12)  # avoid division by zero
        deg_inv_sqrt = deg.pow(-0.5)

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        x = self.linear(x)
        message = x[row] * norm.unsqueeze(1)  # [num_edges, out_channels]
        out = torch.zeros_like(x)
        out = out.index_add(0, col, message)  # aggregate messages

        return out
    

class DrugTargetGNN(nn.Module):
    def __init__(self, node_feature_dim=78, protein_feature_dim=20, hidden_dim=64):
        super().__init__()
        # GNN layers for drug graph
        self.drug_conv1 = GCNConv(node_feature_dim, node_feature_dim)
        self.drug_conv2 = GCNConv(node_feature_dim, node_feature_dim*2)
        self.drug_conv3 = GCNConv(node_feature_dim*2, node_feature_dim*4)
        self.drug_linear1 = nn.Linear(node_feature_dim*4, 1024)
        self.drug_linear2 = nn.Linear(1024, hidden_dim)

        
        #GNN layers for protein graph
        self.protein_conv1 = GCNConv(protein_feature_dim, protein_feature_dim)
        self.protein_conv2 = GCNConv(protein_feature_dim, protein_feature_dim*2)
        self.protein_conv3 = GCNConv(protein_feature_dim*2, protein_feature_dim*4)
        self.protein_linear1 = nn.Linear(protein_feature_dim*4, 1024)
        self.protein_linear2 = nn.Linear(1024, hidden_dim)

        
        # Final layers for combined features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # regression output for pIC50
        )

        
    def forward(self, protein_graph, drug_graph):
        # GNN on drug graph
        x, edge_index,edge_attr = drug_graph.x, drug_graph.edge_index,drug_graph.edge_attr
        p, edge_index_p,edge_attr_p = protein_graph.x, protein_graph.edge_index,protein_graph.edge_attr

        x = F.relu(self.drug_conv1(x, edge_index,edge_attr))
        x = F.relu(self.drug_conv2(x, edge_index,edge_attr))
        x = F.relu(self.drug_conv3(x, edge_index,edge_attr))
        
        x = global_mean_pool(x, drug_graph.batch)  # [batch_size, hidden_dim]
        x = F.relu(self.drug_linear1(x))
        x = F.relu(self.drug_linear2(x))


        p = F.relu(self.protein_conv1(p, edge_index_p,edge_attr_p))
        p = F.relu(self.protein_conv2(p, edge_index_p,edge_attr_p))
        p = F.relu(self.protein_conv3(p, edge_index_p,edge_attr_p))
        
        p = global_mean_pool(p, protein_graph.batch)  # [batch_size, hidden_dim]
        p = F.relu(self.protein_linear1(p))
        p = F.relu(self.protein_linear2(p))



        
        
        
        # Combine embeddings
        combined = torch.cat([x, p], dim=1)
        out = self.final_mlp(combined)
        return out.squeeze()  # [batch_size]