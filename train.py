# coding: utf-8

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from training.ModelTrainer import ModelTrainer, optimizer_setup

from sklearn.model_selection import train_test_split

from models.LSTM import LSTM
from models.RNN import RNN
from models.GRU import GRU

from preprocessing.extraction import FrenchTextExtractor
from preprocessing.extraction import EnglishIMDB
from preprocessing.vectorization import Word2VecVectorizer
from preprocessing.vectorization import OneHotVectorizer
from preprocessing.tokenization import DataCreator

from os.path import dirname, join, abspath

import pickle
import numpy as np

import random

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [encoder] [hyper_parameters]',
                                     description="This program allows to train different models on"
                                                 " different datasets using different encoders. ")
    parser.add_argument('--model', type=str, default="GRU",
                        choices=["LSTM", "RNN", "GRU"])

    parser.add_argument('--datasets_path', type=str, default="./datasets")

    parser.add_argument('--dataset', type=str, default="french-tragedies", choices=["french-tragedies","english-reviews"])

    parser.add_argument('--word_encoding', type=str, default="word2vec", choices=["word2vec","onehot"])

    parser.add_argument('--sequence_size', type=int, default=5, help='The size of the sequences')

    parser.add_argument('--batch_size', type=int, default=20,                        
                            help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num_epochs', type=int, default=20,
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

    root_dir = dirname(abspath(__file__))
    data_dir = join(args.datasets_path)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr

    if args.dataset == "french-tragedies":
        directory = join(data_dir, "livres-en-francais")
        extractor = FrenchTextExtractor()
        saving_path = "saves/livres-en-francais"
    elif args.dataset == "english-reviews":
        directory = join(data_dir, "critiques-imdb")
        extractor = EnglishIMDB()
        saving_path = "saves/english-reviews"
    
    #Extract the sentences
    extractor.index_all_files(directory)
    sentences = extractor.extract_sentences_indexed_files()

    random.seed(0)
    sentences = random.sample(sentences, 30000)

    print("Sentences are extracted !")

    # Vectorize the sentences
    if args.word_encoding == "word2vec":
        vectorizer = Word2VecVectorizer("saves/word2vec.save")
    elif args.word_encoding == "onehot":
        vectorizer = OneHotVectorizer("saves/onehot.save")

    vectorizer.create_vectorization(sentences)

    print("Vectorization computed !")

    sentences = vectorizer.transform_sentences(sentences)

    print("Sentences vectorized !")

    tokenizer = DataCreator(sentences, args.sequence_size, 500000)

    data, labels = tokenizer.tokenize_sentences()

    del sentences

    print("Sequence tokens created !")

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1)

    train_set = train_data, train_labels
    test_set = test_data, test_labels

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'LSTM':
        model = LSTM(input_dim=len(data[0]), hidden_dim=300, output_dim=len(labels[0]))
    elif args.model == 'RNN':
        model = RNN(input_dim=len(data[0]), neurons=30)
    elif args.model == 'GRU':
        model = GRU(input_dim=len(data[0]), hidden_dim=300, output_dim=len(labels[0]))

    model_trainer = ModelTrainer(model=model,
                                data_train=train_set,
                                data_test=test_set,
                                loss_fn=nn.MSELoss(),
                                optimizer_factory=optimizer_factory,
                                batch_size=batch_size,
                                word2vec=vectorizer.model,
                                validation=args.validation,
                                use_cuda=True)

    if args.predict:        
        model_trainer.evaluate_on_test_set()
    
    else:
        print("Entrainement {} sur {} pour {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()

        model_trainer.plot_metrics("saves/fig-"+str(args.model)+"-"+str(args.dataset)+"-"+str(args.sequence_size)+"-"+str(args.batch_size)+".png")

        with open("saves/metrics-"+str(args.model)+"-"+str(args.dataset)+"-"+str(args.sequence_size)+"-"+str(args.batch_size)+".metrics", 'wb') as f:
            pickle.dump(model_trainer.metric_values, f, pickle.HIGHEST_PROTOCOL)

        torch.save(model_trainer.model.state_dict(), "saves/model-"+str(args.model)+"-"+str(args.dataset)+"-"+str(args.sequence_size)+"-"+str(args.batch_size))
