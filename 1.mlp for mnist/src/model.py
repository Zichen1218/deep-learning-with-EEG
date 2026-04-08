import torch.nn as nn
from src.config import CONFIG
from src.config import device

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim//2,output_dim),
        )

    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.model(x)
    
model = MLP(
    input_dim=784,
    hidden_dim = CONFIG['hidden_dim'],
    output_dim=10,
    dropout=CONFIG['dropout'],
).to(device)