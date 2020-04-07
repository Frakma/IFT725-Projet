from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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

class OneHotVector(Vectorizer):
    def __init__(self, save_path):
        self.save_path = save_path

    def create_vectorization(self, sentences):
        words=[]
        for sentence in sentences:
            words=words+sentence
        values = array(words).reshape(-1,1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit_transform(vl)
        self.model = onehot_encoder.fit_transform(values)
        
    
    def save_vectorization(self):
        self.model.save(self.save_path)

    def load_vectorization(self):
        self.model = onehot_encoder.pickle(self.save_path)
        
    def transform_value(self, value):
        return self.model[value]