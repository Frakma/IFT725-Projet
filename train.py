import argparse

import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
from ModelTrainer import ModelTrainer, optimizer_setup

from models.LSTM import LSTM
from models.RNN import RNN
from models.GRU import GRU

#from torchvision import datasets

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [encoder] [hyper_parameters]',
                                     description="This program allows to train different models on"
                                                 " different datasets using different encoders. ")
    parser.add_argument('--model', type=str, default="LSTM",
                        choices=["LSTM", "RNN", "GRU"])

    parser.add_argument('--dataset', type=str, default="Tragédies en français", choices=["Tragédies en français","Critiques de films en anglais",
                    "Articles de Medium", 'Textes religieux et philosophiques', 'Paroles de chansons'])

    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--predict', action='store_true',
                        help="Use model to predict the text")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr

    # Download the data
    data_set_name = args.dataset

    #Apply preprocessing

    #Split to train and test stets
    train_set=[]  # ....
    test_set=[]   # ....

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'LSTM':
        model = LSTM(input_size=1, hidden_layer_size=100, output_size=1) # Rectifier  input_size/output_size
    elif args.model == 'RNN':
        model = RNN(input_size=1, neurons=30)  # Rectifier  input_size
    elif args.model == 'GRU':
        model = GRU()                # Rectifier  Arguments


    model_trainer = ModelTrainer(model=model, data_train=train_test, data_test=test_set)

    if args.predict:        
        model_trainer.evaluate_on_test_set()
    
    else:
        print("Entrainement {} sur {} pour {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()

        #model_trainer.plot_metrics()