import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim=1, neurons=1):
        super().__init__()        
        self.w1 = torch.randn(input_dim, neurons) 
        self.w2 = torch.randn(neurons, neurons)         
        self.b = torch.zeros(1, neurons)
    
    def forward(self, input_seq):
        n=len(input_seq)
        self.Y=[]
        self.Y.append(torch.tanh(torch.mm(input_seq[0], self.w1) + self.b))
        for i in range(1,n):
            self.Y.append(torch.tanh(torch.mm(self.Y[-1], self.w2) + torch.mm(input_seq[i], self.w1) + self.b))

        return self.Y[-1]

"""class RNN(nn.Module):
    def __init__(self, batch_size, input_dim, n_neurons):
        super().__init__()
        
        self.rnn = nn.RNNCell(input_dim, n_neurons)
        self.hx = torch.randn(batch_size, n_neurons) # initialize hidden state
        
    def forward(self, X):
        output = []

        # for each time step
        for i in range(2):
            self.hx = self.rnn(X[i], self.hx)
            output.append(self.hx)
        
        return output, self.hx"""


