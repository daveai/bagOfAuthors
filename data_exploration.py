#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:11:44 2020

@author: d4ve
"""

import nltk
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

sentence = """At eight o'clock on Thursday morning ... Arthur didn't feel very good."""




# Gather all the relative file paths to all the text files provided
# the five and thousand are the txt files to be identified, the other 
# sets are named based on the author.
# Using wild card to make sure we just read txt paths - to avoid possible
# cached data / hidden files or future changes to break the code.
five = glob.glob('500words/*.txt')
thousand = glob.glob('1000words/*.txt')

authors = {
            1: "Almada Negreiros",
            2: "Eca De Queiros",
            3: "Jose Saramago",
            4: "Camilo Castelo Branco",
            5: "Jose Rodrigues Santos",
            6: "Luisa Marques Silva"}

# We created a simple corpus per author, by concatenating all their available
# files together - we simply did this with the following bash command:
# cat *.txt >> corpus.txt

authors_path = {
            1: "trainData/almadaNegreiros/corpus.txt",
            2: "trainData/ecaDeQueiros/corpus.txt",
            3: "trainData/joseSaramago/corpus.txt",
            4: "trainData/camiloCasteloBranco/corpus.txt",
            5: "trainData/joseRodriguesSantos/corpus.txt",
            6: "trainData/luisaMarquesSilva/corpus.txt"}

# Tokenise the author data

vectorizer = TfidfVectorizer()

auth1_tks = vectorizer.fit_transform(txt)


txt = Path(authors_path[1]).read_text()
