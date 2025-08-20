import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Gatted_Attention import GatedAttention
from CNN_Classifier import CNN_Classifier  
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0->human  1-> animals
class ATT_MIL(nn.Module):
    def __init__(self,N_HEAD,ENCODER_N_LAYERS,EMBEDDING_SIZE,INTERMIDIATE_DIM, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ATT_MIL, self).__init__()
        self.M = EMBEDDING_SIZE
        self.L = INTERMIDIATE_DIM
        self.ENCODER_N_LAYERS=ENCODER_N_LAYERS
        self.ATTENTION_BRANCHES = 1
        self.N_HEAD=N_HEAD
        self.device = device

        # embedding 
        self.encoder = Encoder(N_HEAD, ENCODER_N_LAYERS, EMBEDDING_SIZE)
        # instance level 
        self.instance_level_attention = GatedAttention(self.M, self.L)
        # bag level 
        self.bag_level_attention = GatedAttention(self.M, self.L)
        # classifier
        self.classifier = CNN_Classifier(self.M)

    
    def forward(self, datas,ids,Seq_ids):
        A_vec_2_all=[]
        A_vec_all=[]
        #### STEP 1:embeddings
        datas = datas.float()  
        instances=self.transformer_encoder(datas) 
        
        #### STEP 2: INSTANCE-LEVEL ATTENTION ####
        W_1= self.instance_level_attention(instances)  
        inner_bags = torch.unique_consecutive(Seq_ids)
        output = torch.empty(((len(inner_bags), self.M))).to(self.device)
        super_ids = torch.empty(((len(inner_bags))))
        for i, bag in enumerate(inner_bags):
            A_vec=F.softmax(W_1[0][Seq_ids == bag],dim=0)
            A_vec_all.append(A_vec)
            output[i] = torch.matmul(A_vec, instances[Seq_ids == bag])
            super_ids[i]=ids[Seq_ids == bag][0]
        
        ### STEP 3: BAG-LEVEL ATTENTION ####
        W_2 = self.bag_level_attention(output)  
        outer_bags = torch.unique_consecutive(super_ids)
        output2 = torch.empty(((len(outer_bags), self.M))).to(self.device)
        for i, bag in enumerate(outer_bags):
            A_vec_2=F.softmax(W_2[0][super_ids == bag],dim=0)
            A_vec_2_all.append(A_vec_2)
            output2[i] = torch.matmul(A_vec_2, output[super_ids == bag])

        
        ### STEP 4: CLASSIFICATION ####
        output2 = output2.unsqueeze(1)
        Y_prob = self.classifier(output2)  # Shape: [batch_size, 1]
        Y_hat = torch.ge(Y_prob, 0.5).float() 
        return Y_prob, Y_hat, A_vec_all, A_vec_2_all,output2
    
    
