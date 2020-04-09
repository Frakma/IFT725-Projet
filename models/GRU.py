import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=100, output_dim=100):
        super().__init__()
        self.hidden_layer_size = hidden_dim
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq):
        gru_out,_ = self.gru(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(gru_out.view(len(input_seq), -1))
        return predictions
