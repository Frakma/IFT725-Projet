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
        vectorized_sentences = []
        for sentence in sentences:
            vectorized_sentence = []
            for word in sentence:
                vectorized_sentence.append(self.transform_value(word))

            vectorized_sentences.append(vectorized_sentences)

        return vectorized_sentences

class Word2VecVectorizer(Vectorizer):
    def __init__(self, save_path):
        self.save_path = save_path

    def create_vectorization(self, sentences):
        self.model = Word2Vec(sentences, min_count=1, workers=4)
    
    def save_vectorization(self):
        self.model.save(self.save_path)

    def load_vectorization(self):
        self.model = Word2Vec.load(self.save_path)
        
    def transform_value(self, value):
        return self.model[value]
