import torch
import torch.nn as nn

#https://blog.floydhub.com/gru-with-pytorch/
class GRU(nn.Module):
    def __init__(self, input_dim=500, hidden_layer_size=100, output_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        self.gru = nn.GRU(input_dim, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        #self.relu = nn.ReLU()
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
    
    def forward(self, input_seq):
        gru_out, self.hidden_cell = self.gru(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(gru_out.view(len(input_seq), -1))
        return predictions