import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, neurons=1):
        super().__init__()        
        self.w1 = torch.randn(input_size, neurons) 
        self.w2 = torch.randn(neurons, neurons)         
        self.b = torch.zeros(1, neurons)
    
    def forward(self, text):
        n=len(text)
        self.Y=[]
        self.Y.append(torch.tanh(torch.mm(text[0], self.w1) + self.b))
        for seq in range(1,n):
            self.Y.append(torch.tanh(torch.mm(self.Y[-1], self.w2) + torch.mm(text[seq], self.w1) + self.b))

        return self.Y
