import numpy as np
import pandas as pd
from os import listdir
from os.path import dirname, abspath, join, isfile, isdir
import re
import string 
import pickle

char_to_remove = "—«»\t"
words_to_replace = {
    "M. ":"Monsieur "
}
char_end_sentence = ".!?"

last_intro_sequence = "Vous devez attribuer l’œuvre aux différents auteurs, y compris à Bibebook."

root_dir = dirname(dirname(abspath(__file__)))
data_dir = join(root_dir,"datasets")
frenchbooks_dir = join(data_dir, "livres-en-francais")

def format_txt_book(path):
    book_file = open(path, encoding="utf-8")

    # enlever le début
    formatting = book_file.read()
    formatting= formatting[formatting.rfind(last_intro_sequence)+len(last_intro_sequence):]

    # enlever les caractères indésirables
    formatting = re.sub('['+char_to_remove+']','', formatting)
    # enlever les sauts de lignes
    formatting = re.sub('\n+',' ', formatting)
    # enlever les doubles espaces
    formatting = re.sub(' +', ' ', formatting)
    # remplacer les suites de caractères qui peuvent poser problème
    for word in words_to_replace:
        formatting = re.sub(word,words_to_replace[word], formatting)

    return formatting

authors = [f for f in listdir(frenchbooks_dir) if isdir(join(frenchbooks_dir, f))]

french_books_data = []

for author in authors:
    author_dir = join(frenchbooks_dir, author)

    books = [f for f in listdir(author_dir) if isdir(join(author_dir, f))]

    for book in books:
        book_dir = join(author_dir, book)

        txt_path = None   

        for file in listdir(book_dir):
            if file.endswith(".txt"):
                txt_path = join(book_dir,file)
                french_books_data.append([author, book, txt_path])

french_books_dataframe = pd.DataFrame(french_books_data, columns=["Author","Book","Path"])

all_sentences = []

for index, row in french_books_dataframe.iterrows():
    print(row["Author"], "- ", row["Book"])
    formatted = format_txt_book(row["Path"])

    # changer le regex pour prendre en compte qu'avec des ! ou ? il y a un espace avant
    sentences = re.sub('['+char_end_sentence+'] ', '\n', formatted).split('\n')
    sentences = [sentence.split() for sentence in sentences]

    all_sentences.extend(sentences)

print("Nous avons pu extraire",len(all_sentences), "phrases")

with open("saves/french_sentences.save", 'wb') as f:
    pickle.dump(sentences, f)