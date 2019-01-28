#!/usr/bin/env python2

from lib.utilities import read_dir
from lib.data import DataFormatter
from lib.utilities import load_doc_list
import string


"""
    DATA_PATH: Path of directory storing all the documents
    TEMP_PATH: Path of directory storing transient results
    STOPWORDS_DIR: Path of directory storing stopwords, stopwords can only be in a csv file, each line is a stopword
    BUSINESSWORDS_DIR: Similar to STOPWORDS_DIR, the directory path of business words 
    
"""


DATA_PATH = "./data/cleantext"
TEMP_PATH = "./temp"
STOPWORDS_DIR = "./data/stopwords"
BUSINESSWORDS_DIR = "./data/businesswords"


def clean_text(content):
    return content.translate(
        {ord(c): None for c in string.punctuation}
    ).lower()


def get_doc_paths(data_path):
    doc_set = set(load_doc_list())
    doc_paths = read_dir(data_path, ends=".txt")
    return [(doc[:-4], path) for doc, path in doc_paths if doc[:-4] in doc_set]

if __name__ == "__main__":

    # get the path of all documents
    doc_paths = get_doc_paths(DATA_PATH)

    # create data formatter
    formatter = DataFormatter(TEMP_PATH, STOPWORDS_DIR, BUSINESSWORDS_DIR)

    # generate gram directly from document, clean_text is the function you want to preprocess the document
    formatter.generate_grams_from_raw_doc(doc_paths, clean_text)

    # generate all training data and testing data
    formatter.generate_all(train_ratio=0.7)

