import torch
import torch.nn as nn

#https://discuss.pytorch.org/t/nan-loss-in-rnn-model/655

class RNN(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=300, output_dim=100):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq):
        hidden=torch.zeros(input_seq.shape[0], self.hidden_dim)
        combined = torch.cat((input_seq, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output