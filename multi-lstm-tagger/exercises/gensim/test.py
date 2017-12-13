#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load('/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/journal/jpl/w2vec/word2vec-jp/word2vec.gensim.model')
print word_vectors[u'計算']