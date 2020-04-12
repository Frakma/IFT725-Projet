from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
from abc import ABC, abstractmethod 

class Vectorizer(ABC):
    """Abstract class which needs to be inherited for each encoding
    """
    def create_vectorization(self, sentences):
        """Function to compute the vectorization

            Parameters:
            sentences (list): List of list of words
        """
        pass

    def save_vectorization(self):
        """Function which save the vectorization object into disk
        """
        pass

    def load_vectorization(self):
        """Function which load the vectorization object from disk
        """
        pass

    def transform_value(self, value):
        """Function which transform a word into a vector

            Parameters:
            value (string): word to transform

            Returns:
            array:Vectorized word
        """
        pass

    def transform_sentences(self, sentences):
        """Function which transform a list of word-splitted sentences

            Parameters:
            sentences (list): list of word-splitted sentences

            Returns:
            list:list of sentences with vectorized words
        """
        
        vectorized_sentences = []
        for sentence in sentences:
            result = map(self.transform_value, sentence)
            vectorized_sentences.append(list(result))

        return vectorized_sentences

class Word2VecVectorizer(Vectorizer):
    """Class to implement Word2Vec vectorizer"""
    def __init__(self, save_path):
        self.save_path = save_path
        self.index_word = dict()
    
    def save_vectorization(self):
        self.model.save(self.save_path)

    def load_vectorization(self):
        self.model = Word2Vec.load(self.save_path)
        
    def transform_value(self, value):
        return self.index_word[value]

    def create_vectorization(self, sentences):
        self.model = Word2Vec(sentences, min_count=1)

        self.weights = self.model.wv.vectors

        for i, word in enumerate(self.model.wv.vocab):
            self.index_word[word] = i

class OneHotVectorizer(Vectorizer):
    """Class to implement Onhotencoding"""
    def __init__(self, save_path):
        self.save_path = save_path

    def create_vectorization(self, sentences):
        values=[]
        for i in range(len(sentences)):
            values+=sentences[i]
        values=np.array(values).reshape(-1,1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(values)
        self.model = onehot_encoder
    
    def save_vectorization(self):
        pickle.dump(self.model, open(self.save_path, 'wb'))

    def load_vectorization(self):
        self.model = pickle.load(open(self.save_path, 'rb'))
        
    def transform_sentences(self, sentences):
        values=[]
        for i in range(len(sentences)):
            values+=sentences[i]
        values=np.array(values).reshape(-1,1)
        return self.model.transform(values)