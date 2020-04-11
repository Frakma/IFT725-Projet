import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class GRU(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, sequence_size, dropout=0.2):
        super().__init__() 
        nums_embedding, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(weights_matrix))
        self.embedding.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(hidden_size*sequence_size, nums_embedding)

        self.sequence_size = sequence_size
        
    def forward(self, inputs):   #, hidden):
        x = self.embedding(inputs) 

        #x , hidden = self.gru(x, hidden)
        x, _ = self.gru(x)    #, hidden)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)       #, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

class LSTM(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, sequence_size, dropout=0.2):
        super().__init__() 
        nums_embedding, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(weights_matrix))
        self.embedding.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size*sequence_size, nums_embedding)

        self.sequence_size = sequence_size
        
    def forward(self, inputs):   #, hidden):
        x = self.embedding(inputs) 

        #x , hidden = self.lstm(x, hidden)
        x, _ = self.lstm(x)    #, hidden)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)       #, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))