import torch
import torch.nn as nn

#https://blog.floydhub.com/gru-with-pytorch/
class GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1,  drop_prob=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        #self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim,  batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h