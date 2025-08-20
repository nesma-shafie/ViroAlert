import torch.nn as nn
from CONSTANTS import  max_smile_len
##real->1  && fake->0
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, conv_filters=3, kernel_size=3, sequence_length=max_smile_len):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, conv_filters, 5, padding=2)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, 7, padding=3)
        self.relu = nn.ReLU()
        # Compute flattened size
        self.flattened_size = (conv_filters * 2) * sequence_length
        self.fc = nn.Linear(self.flattened_size, 1)
        self.dropout = nn.Dropout(0.3)  # Dropout layer
    
    def forward(self, x):
        x = self.embedding(x)                     # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)                    # (batch_size, embedding_dim, sequence_length)

        x = self.relu(self.conv1(x))              # (batch_size, conv_filters, sequence_length)
        x = self.relu(self.conv2(x))              # (batch_size, conv_filters*2, sequence_length)
        x = self.dropout(x)                       # Dropout after conv2

        x = x.view(x.size(0), -1)                 # Flatten
        x = self.fc(x)                            # (batch_size, 1)
        
        return x
