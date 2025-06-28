from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    