#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:47:40 2020

@author: d4ve
"""
import nltk
import glob
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


tokenised_paths = glob.glob('trainData/*/*.txt')

f1 = open(tokenised_paths[0])
f1 = f1.read()
f2 = open(tokenised_paths[44])
f2 = f2.read()

corpus = [f1, f2]

vocabulary = ['de', 'e', 'te', 'larga', 'tambem']

pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)

pipe['count'].transform(corpus).toarray()

pipe['tfid'].idf_

pipe.transform(corpus).shape