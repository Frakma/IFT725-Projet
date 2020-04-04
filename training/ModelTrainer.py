import torch

class ModelTrainer(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def train(self, num_epochs):
        pass