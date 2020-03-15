#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:11:44 2020

@author: d4ve
"""

import nltk
import glob
from sklearn.feature_extraction.text import TfidfVectorizer


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