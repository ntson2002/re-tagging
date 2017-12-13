# -*- coding: utf-8 -*-

import fasttext
model = fasttext.load_model('/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/journal/jpl/w2vec/wiki.ja.vec')
print model.words # list of words in dictionary
print model[u'申出'] # get the vector of the word 'king'