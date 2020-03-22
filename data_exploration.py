#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:11:44 2020

@author: d4ve
"""

import nltk
import glob
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


# Gather all the relative file paths to all the text files provided
# the five and thousand are the txt files to be identified, the other 
# sets are named based on the author.
# Using wild card to make sure we just read txt paths - to avoid possible
# cached data / hidden files or future changes to break the code.
five = glob.glob('500words/*.txt')
thousand = glob.glob('1000words/*.txt')

tokenised_paths = glob.glob('trainData/*/*tokenised.csv')

# Custom Variables
authors = {
            1: "almadaNegreiros",
            2: "ecaDeQueiros",
            3: "joseSaramago",
            4: "camiloCasteloBranco",
            5: "joseRodriguesSantos",
            6: "luisaMarquesSilva"}

df = pd.DataFrame()

# NLTK Tools
stopwords = list(nltk.corpus.stopwords.words('portuguese'))
stemmer = nltk.stem.RSLPStemmer()

# Function that will map an author given the path of the file, to the key of
# our authors dictionary.
def map_author(path):
    for key, author in authors.items():
        if author in path:
            return key
# Given a dataframe, stem_df will stem all columns and return a df with stemmed
# columns and their values summed together.
def stem_df(df):
    stemmed = [stemmer.stem(str(i)) for i in df.columns]
    df.columns = stemmed
    df = df.groupby(level=0, axis=1).sum()
    return df

# For each document, read it into pandas, remove stopwords and transpose the matrix.
# Then clean the dataframe up and merge it onto the main df.
for path in tokenised_paths:
    temp_df = pd.read_csv(path, header=None)
    temp_df = temp_df[~temp_df[1].isin(stopwords)].T
    temp_df.columns = temp_df.iloc[1]
    temp_df.drop(1, inplace=True)
    temp_df = stem_df(temp_df)
    temp_df['author'] = map_author(path)
    df = pd.concat([df, temp_df])

df.reset_index(drop=True, inplace=True)

df.to_csv("corpus_word_tokenised.csv")

df_auth = df.groupby(['author']).sum()

##############################################################################


    