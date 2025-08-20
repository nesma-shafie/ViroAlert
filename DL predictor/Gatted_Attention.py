import torch
import torch.nn as nn

class GatedAttention(nn.Module):
    def __init__(self, M, L):
        super(GatedAttention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = 1
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )
        self.attention_W = nn.Linear(self.L, self.ATTENTION_BRANCHES) 
        
    def forward(self, instances):
        V = self.attention_V(instances)  
        U = self.attention_U(instances)  
        W = self.attention_W(V * U)
        W = torch.transpose(W, 1, 0)  
        return W
        
        