from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from abc import ABC, abstractmethod 

class Vectorizer(ABC):
    def create_vectorization(self, sentences):
        pass

    def save_vectorization(self):
        pass

    def load_vectorization(self):
        pass

    def transform_value(self, value):
        pass

    def transform_sentences(self, sentences):
        pass

class Word2VecVectorizer(Vectorizer):
    def __init__(self, save_path):
        self.save_path = save_path

    def create_vectorization(self, sentences):
        wordvecmodel = Word2Vec(sentences, min_count=2)
        self.wv = wordvecmodel.wv
    
    def save_vectorization(self):
        with open(self.save_path, "wb") as f:
            self.wv.save(f)

    def load_vectorization(self):
        self.wv = KeyedVectors.load(self.save_path, mmap='r')
        
    def transform_value(self, value):
        return self.wv[value]

    def transform_sentences(self, sentences):
        pass