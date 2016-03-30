# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:14:31 2016

@author: chuckgu
"""

from gensim.models import Word2Vec
model = Word2Vec.load_word2vec_format('/home/chuckgu/Desktop/project/Alphabeta/data/GoogleNews-vectors-negative300.bin', binary=True)
print model.similarity('wear', 'shoe')