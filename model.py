from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.data import Data




torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-detect GPU




# 0->human  1-> animals
class GatedAttention(nn.Module):
    def __init__(self,N_HEAD,ENCODER_N_LAYERS,EMBEDDING_SIZE,INTERMIDIATE_DIM):
        super(GatedAttention, self).__init__()
        self.M = EMBEDDING_SIZE
        self.L = INTERMIDIATE_DIM
        self.ENCODER_N_LAYERS=ENCODER_N_LAYERS
        self.ATTENTION_BRANCHES = 1
        self.N_HEAD=N_HEAD

        # embedding 
        self.encoder_layer = TransformerEncoderLayer(d_model=self.M, nhead=self.N_HEAD)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.ENCODER_N_LAYERS)
        
        # instance level 
        self.attention_V_1 = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U_1 = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w_1 = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


        # bag level 
        self.attention_V_2 = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U_2 = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w_2 = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


        
        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=self.ATTENTION_BRANCHES, out_channels=128, kernel_size=4, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Flatten(),  # Converts to 1D before fully connected layers
            nn.Linear(128 * ((self.M) // 4), 256),  # Adjust size based on sequence length
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    
    def forward(self, datas,ids,Seq_ids):
        A_vec_2_all=[]
        A_vec_all=[]
        #### STEP 1:embeddings
        datas = datas.float()  # Ensure correct dtype
        instances=self.transformer_encoder(datas) 
        
        #### STEP 2: INSTANCE-LEVEL ATTENTION ####
        # Apply attention mechanisms per bag (over instances_per_bag)
        A_V = self.attention_V_1(instances)  
        A_U = self.attention_U_1(instances)  
        A = self.attention_w_1(A_V * A_U)
        A = torch.transpose(A, 1, 0)  
        inner_bags = torch.unique_consecutive(Seq_ids)
      
        output = torch.empty(((len(inner_bags), self.M))).to(device)
        super_ids = torch.empty(((len(inner_bags))))
        for i, bag in enumerate(inner_bags):
            A_vec=F.softmax(A[0][Seq_ids == bag],dim=0)
            A_vec_all.append(A_vec)
            output[i] = torch.matmul(A_vec, instances[Seq_ids == bag])
            super_ids[i]=ids[Seq_ids == bag][0]
        
        ### STEP 3: BAG-LEVEL ATTENTION ####
        A_V_2 = self.attention_V_2(output)  
        A_U_2 = self.attention_U_2(output)  
        A_2 = self.attention_w_2(A_V_2 * A_U_2)  
        A_2 = torch.transpose(A_2, 1,0)   

      
        outer_bags = torch.unique_consecutive(super_ids)
        output2 = torch.empty(((len(outer_bags), self.M))).to(device)

        for i, bag in enumerate(outer_bags):
            A_vec_2=F.softmax(A_2[0][super_ids == bag],dim=0)
            A_vec_2_all.append(A_vec_2)
            output2[i] = torch.matmul(A_vec_2, output[super_ids == bag])

        
        
        ### STEP 4: CLASSIFICATION ####
        # output2 = output2.view(output2.shape[0], -1)  # Flatten over bags_per_bag for classification
        output2 = output2.unsqueeze(1)  # Add a channel dimension


        Y_prob = self.classifier(output2)  # Shape: [batch_size, 1]
        Y_hat = torch.ge(Y_prob, 0.5).float()  # Convert probabilities to binary predictions
        return Y_prob, Y_hat, A_vec_all, A_vec_2_all
    
def custom_collate(batch):
    protein_graphs = ([item[0] for item in batch])  
    drug_graphs = [item[1] for item in batch]                 

    batch_protein_graphs = Batch.from_data_list(protein_graphs)     # Combine graphs into a single batched graph
    batch_drug_graphs = Batch.from_data_list(drug_graphs)     # Combine graphs into a single batched graph

    return batch_protein_graphs, batch_drug_graphs

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
    
class DrugProteinDataset(torch.utils.data.Dataset):
    def __init__(self, protein_graphs, drug_graphs):
        self.protein_graphs = protein_graphs
        self.drug_graphs = drug_graphs
    
    def __len__(self):
        return len(self.drug_graphs)
    
    def __getitem__(self, idx):
        protein_graphs = protein_graph_to_data(self.protein_graphs[idx])
        drug_graph = drug_graph_to_data(self.drug_graphs[idx])
        return protein_graphs, drug_graph


    
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
    
