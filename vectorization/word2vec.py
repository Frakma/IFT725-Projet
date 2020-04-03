from gensim.models import Word2Vec
import torch
import pytorch.nn as nn
import pickle

with open("saves/french_sentences.save", "rb") as f:
    french_sentences = pickle.load(f)

word2vec = Word2Vec(french_sentences, min_count=2)

print(word2vec.wv.most_similar('monsieur'))

# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/ 
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

