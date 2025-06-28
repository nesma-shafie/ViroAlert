import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool       

def drug_graph_to_data(drug_graph):
    mol_size, nodes, edges, edges_type = drug_graph
    x = torch.tensor(nodes, dtype=torch.float)  # [num_nodes, node_features]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edges_type, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def protein_graph_to_data(protein_graph):
    node_features,edge_index,edge_attr = protein_graph
    x = node_features
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

class DrugProteinDataset(torch.utils.data.Dataset):
    def __init__(self, protein_graphs, drug_graphs, pIC50_values):
        self.protein_graphs = protein_graphs
        self.drug_graphs = drug_graphs
        self.pIC50_values = pIC50_values
    
    def __len__(self):
        return len(self.pIC50_values)
    
    def __getitem__(self, idx):
        protein_graphs = protein_graph_to_data(self.protein_graphs[idx])
        drug_graph = drug_graph_to_data(self.drug_graphs[idx])
        pIC50_value = torch.tensor(self.pIC50_values[idx], dtype=torch.float)
        return protein_graphs, drug_graph, pIC50_value

def custom_collate(batch):
    protein_graphs = ([item[0] for item in batch])  
    drug_graphs = [item[1] for item in batch]                 
    labels = torch.stack([item[2] for item in batch])        

    batch_protein_graphs = Batch.from_data_list(protein_graphs)     
    batch_drug_graphs = Batch.from_data_list(drug_graphs)     

    return batch_protein_graphs, batch_drug_graphs, labels

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

        # Normalization 
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
    def __init__(self, node_feature_dim=78, protein_feature_dim=20, hidden_dim=128):
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