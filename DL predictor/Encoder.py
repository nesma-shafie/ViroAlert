from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, N_HEAD, ENCODER_N_LAYERS, EMBEDDING_SIZE):
        super(Encoder, self).__init__()
        self.M = EMBEDDING_SIZE
        self.ENCODER_N_LAYERS = ENCODER_N_LAYERS
        self.N_HEAD = N_HEAD

        self.encoder_layer = TransformerEncoderLayer(d_model=self.M, nhead=self.N_HEAD)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.ENCODER_N_LAYERS)

    def forward(self, datas):
        datas = datas.float()  
        datas = self.transformer_encoder(datas)
        return datas  
            
