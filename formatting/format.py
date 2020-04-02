import numpy as np
import pandas as pd
from os import listdir
from os.path import dirname, abspath, join, isfile, isdir
import re
import string 
char_to_remove = "—«»\t"
words_to_replace = {
    "M. ":"Monsieur "
}
char_end_sentence = ".!?"

root_dir = dirname(dirname(abspath(__file__)))
data_dir = join(root_dir,"datasets")
frenchbooks_dir = join(data_dir, "livres-en-francais")

def format_txt_book(path):
    book_file = open(path, encoding="utf-8")

    # enlever le début 

    # enlever les caractères indésirables
    formatting = re.sub('['+char_to_remove+']','', book_file.read())
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

formatted = format_txt_book(french_books_dataframe["Path"][0])

# changer le regex pour prendre en compte qu'avec des ! ou ? il y a un espace avant
sentences = re.sub('['+char_end_sentence+'] ', '\n', formatted).split('\n') 

print(sentences[:50])