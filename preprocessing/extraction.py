import numpy as np
import pandas as pd
from os import listdir
from os.path import dirname, abspath, join, isfile, isdir
import re
import string 

from abc import ABC, abstractmethod

class TextExtractor(ABC):
    def index_all_files(self, root_dir):
        pass

    def load_and_clean_file(self, path):
        pass

    def extract_sentences_from_text(self, text_data):
        pass

    def extract_sentences_indexed_files(self):
        pass

class FrenchTextExtractor(TextExtractor):
    def __init__(self):
        self.last_intro_sequence = "Vous devez attribuer l’œuvre aux différents auteurs, y compris à Bibebook."
        self.end_sequence_chars = ".!?"
        self.words_to_replace = {
            "M. " : "Monsieur "
        }
        self.to_remove_chars = "—«»\t"

    def index_all_files(self, root_dir):
        french_books_data = []

        authors = [f for f in listdir(root_dir) if isdir(join(root_dir, f))]

        for author in authors:
            author_dir = join(root_dir, author)

            books = [f for f in listdir(author_dir) if isdir(join(author_dir, f))]

            for book in books:
                book_dir = join(author_dir, book)

                txt_path = None   

                for file in listdir(book_dir):
                    if file.endswith(".txt"):
                        txt_path = join(book_dir,file)
                        french_books_data.append([author, book, txt_path])

        self.df_files = pd.DataFrame(french_books_data, columns=["author","book","path"])

    def trim_beginning(self, text_data):
        return text_data[text_data.rfind(self.last_intro_sequence)+len(self.last_intro_sequence):]

    def load_and_clean_file(self, path):
        book_file = open(path, encoding="utf-8")
        formatting = book_file.read()

        # enlever le début
        formatting = self.trim_beginning(formatting)

        # enlever les caractères indésirables
        formatting = re.sub('['+self.to_remove_chars+']','', formatting)
        # enlever les sauts de lignes
        formatting = re.sub('\n+',' ', formatting)
        # enlever les doubles espaces
        formatting = re.sub(' +', ' ', formatting)
        # remplacer les suites de caractères qui peuvent poser problème
        for word in self.words_to_replace:
            formatting = re.sub(word, self.words_to_replace[word], formatting)

        return formatting

    def extract_sentences_from_text(self, text_data):
        # changer le regex pour prendre en compte qu'avec des ! ou ? il y a un espace avant
        sentences = re.sub('['+self.end_sequence_chars+'] ', '\n', text_data).split('\n')

        splitter = re.compile("[, ;.()]")
        sentences = [splitter.split(sentence) for sentence in sentences]

        return sentences

    def extract_sentences_indexed_files(self):
        all_sentences = []

        for index, row in self.df_files.iterrows():
            print("En train d'extraire les phrases du livre :",row["author"], "-", row["book"])

            file_data = self.load_and_clean_file(row["path"])

            sentences = self.extract_sentences_from_text(file_data)

            all_sentences.extend(sentences)

        return all_sentences

class EnglishIMDB(TextExtractor):
    def __init__(self):
        self.end_sequence_chars = ".!?"
        self.words_to_replace = {
            "&":"and"
        }
        self.to_remove_chars = "«»\t\""

    def index_all_files(self, root_dir):
        self.file = join(root_dir,"IMDB_Dataset.csv")

    def trim_beginning(self, text_data):
        pass

    def load_and_clean_file(self, path):
        dataset = pd.read_csv(self.file)

        formatting = ' '.join(dataset["review"].values)

        formatting = re.sub('['+self.to_remove_chars+']','', formatting)
        formatting = re.sub('<.*?>','', formatting)

        for word in self.words_to_replace:
            formatting = re.sub(word, self.words_to_replace[word], formatting)

        return formatting

    def extract_sentences_from_text(self, text_data):
        # changer le regex pour prendre en compte qu'avec des ! ou ? il y a un espace avant
        sentences = re.sub('['+self.end_sequence_chars+']', '\n', text_data).split('\n')

        splitter = re.compile("[, -;.()]")

        sentences = [splitter.split(sentence) for sentence in sentences]

        removed_empty = []
        for sentence in sentences:
            removed_empty.append([x for x in sentence if len(x) > 0])

        sentences = [x for x in removed_empty if len(x) > 0]

        return sentences

    def extract_sentences_indexed_files(self):
        data = self.load_and_clean_file(self.file)

        sentences = self.extract_sentences_from_text(data)

        return sentences
