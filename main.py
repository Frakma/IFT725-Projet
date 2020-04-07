from preprocessing.extraction import FrenchTextExtractor
from preprocessing.extraction import EnglishIMDB
from preprocessing.vectorization import Word2VecVectorizer
from preprocessing.tokenization import DataCreator

from os.path import dirname, join, abspath
import pickle

root_dir = dirname(abspath(__file__))

data_dir = join(root_dir,"datasets")

save_french_sentences_path = "saves/french_sentences.save"

# ## NOTE : à décommenter lorsqu'on ne charge pas des données existantes
# frenchbooks_dir = join(data_dir, "livres-en-francais")

# text_extractor = FrenchTextExtractor()
# text_extractor.index_all_files(frenchbooks_dir)

# extracted_sentences = text_extractor.extract_sentences_indexed_files()
# print("Nous avons pu extraire", len(extracted_sentences), "phrases")

# with open(save_french_sentences_path, 'wb') as f:
#     pickle.dump(extracted_sentences, f)
# ##

# ## NOTE : à décommenter lorsqu'on charge des données existantes
# with open(save_french_sentences_path, "rb") as f:
#     extracted_sentences = pickle.load(f)
# ##

# vectorizer = Word2VecVectorizer("saves/french_word2vec.save")

# ## NOTE : à décommenter lorsqu'on ne charge pas des données existantes
# vectorizer.create_vectorization(extracted_sentences)
# vectorizer.save_vectorization()
# ###

# ## NOTE : à décommenter lorsqu'on charge des données existantes
# vectorizer.load_vectorization()
# ##

# Pour tester l'encodage Word2Vec
# print(vectorizer.wv.most_similar("dormir"))

moviereviews_dir = join(data_dir, "critiques-imdb")
text_extractor = EnglishIMDB()
text_extractor.index_all_files(moviereviews_dir)
sentences = text_extractor.extract_sentences_indexed_files()
print("Nous avons pu extraire", len(sentences), "phrases")

print(sentences[:50])

vectorizer = Word2VecVectorizer("saves/reviews_word2vec.save")
vectorizer.load_vectorization()

vectorizer = vectorizer.transform_sentences(sentences) 