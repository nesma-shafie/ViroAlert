import torch
import torch.nn as nn   
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from graph_utils import GCNConv  

def drug_graph_to_data(drug_graph):
    mol_size, nodes, edges, edges_type = drug_graph
    x = torch.tensor(nodes, dtype=torch.float)  # [num_nodes, node_features]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edges_type, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

class DrugProteinDataset(torch.utils.data.Dataset):
    def __init__(self, protein_features, drug_graphs, pIC50_values):
        self.protein_features = protein_features
        self.drug_graphs = drug_graphs
        self.pIC50_values = pIC50_values
    
    def __len__(self):
        return len(self.pIC50_values)
    
    def __getitem__(self, idx):
        protein_feature = torch.tensor(self.protein_features[idx], dtype=torch.float)
        drug_graph = drug_graph_to_data(self.drug_graphs[idx])
        pIC50_value = torch.tensor(self.pIC50_values[idx], dtype=torch.float)
        return protein_feature, drug_graph, pIC50_value

def custom_collate(batch):
    protein_feats = torch.stack([item[0] for item in batch])  # [batch_size, protein_feature_dim]
    drug_graphs = [item[1] for item in batch]                 
    labels = torch.stack([item[2] for item in batch])         # [batch_size]

    batch_drug_graphs = Batch.from_data_list(drug_graphs)     

    return protein_feats, batch_drug_graphs, labels

class DrugTargetGNN(nn.Module):
    def __init__(self, node_feature_dim=78, protein_feature_dim=50, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, node_feature_dim)
        self.conv2 = GCNConv(node_feature_dim, node_feature_dim*2)
        self.conv3 = GCNConv(node_feature_dim*2, node_feature_dim*4)

        
        self.lineargraph = nn.Linear(node_feature_dim*4, hidden_dim)
        
        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_feature_dim, protein_feature_dim*2),
            nn.ReLU(),
            nn.Linear(protein_feature_dim*2, protein_feature_dim*4),
            nn.ReLU(),
            nn.Linear(protein_feature_dim*4, hidden_dim)
            
        )
        
        #  combined features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  # regression output for pIC50
        )

        
    def forward(self, protein_feat, drug_graph):
        x, edge_index,edge_attr = drug_graph.x, drug_graph.edge_index,drug_graph.edge_attr
        x = F.relu(self.conv1(x, edge_index,edge_attr))
        x = F.relu(self.conv2(x, edge_index,edge_attr))
        x = F.relu(self.conv3(x, edge_index,edge_attr))
        
        x = global_mean_pool(x, drug_graph.batch)  # [batch_size, hidden_dim]
        x = F.relu(self.lineargraph(x))

        
        # protein feature embedding
        p = self.protein_mlp(protein_feat)  # [batch_size, hidden_dim]
        
        # combine embeddings
        combined = torch.cat([x, p], dim=1)
        out = self.final_mlp(combined)
        return out.squeeze()  # [batch_size]