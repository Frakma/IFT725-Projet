import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, neurons=1):
        super().__init__()        
        self.w1 = torch.randn(input_size, neurons) 
        self.w2 = torch.randn(neurons, neurons)         
        self.b = torch.zeros(1, neurons)
    
    def forward(self, input_seq):
        n=len(input_seq)
        self.Y=[]
        self.Y.append(torch.tanh(torch.mm(input_seq[0], self.w1) + self.b))
        for i in range(1,n):
            self.Y.append(torch.tanh(torch.mm(self.Y[-1], self.w2) + torch.mm(input_seq[i], self.w1) + self.b))

        return self.Y[-1]
