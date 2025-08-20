import torch.nn as nn

class CNN_Classifier(nn.Module):
    def __init__(self, M):
        super(CNN_Classifier, self).__init__()
        self.M = M
        self.ATTENTION_BRANCHES = 1  # Number of attention branches

        # Define the CNN classifier
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
            nn.Flatten(),  
            nn.Linear(128 * ((self.M) // 4), 256),  
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        x=self.classifier(x)
        return x

