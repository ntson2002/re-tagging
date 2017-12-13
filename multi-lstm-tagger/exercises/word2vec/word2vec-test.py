#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gensim

# model = gensim.models.Word2Vec.load_word2vec_format('/home/s1520203/programs/word2vec/output/du-lieu-luat-all-tokenized.dat-100.bin', binary=True, unicode_errors='ignore')
# model = gensim.models.Word2Vec.load_word2vec_format('/home/s1520203/programs/word2vec/output/tokenize.vi.256.bin', binary=True, unicode_errors='ignore')
# model = gensim.models.KeyedVectors.load_word2vec_format('/home/s1520203/programs/word2vec/output/all-data-tokenized.vi-100.bin', binary=True, unicode_errors='ignore')
from datetime import datetime
print str(datetime.now())
model = gensim.models.KeyedVectors.load_word2vec_format('/home/sonnguyen/jaist/Bitbucket/lstm-crf-tagging/experiments/journal/jpl/w2vec/wiki.ja.vec', binary=False, unicode_errors='ignore')
print str(datetime.now())
# model = gensim.models.Word2Vec.load_word2vec_format('/work/sonnguyen/glove/glove_word2vec/glove.6B.100d.w2vec', binary=False, unicode_errors='ignore')
# model = gensim.models.Word2Vec.load_word2vec_format('/work/sonnguyen/glove/glove_word2vec/glove.twitter.27B.100d.w2vec', binary=False, unicode_errors='ignore')
# print len(model)

w = model[u"より"]
print type(w)
print len(w)
# print model[u"申出"]
print model[u"計算"]
# print model.similarity(u'đường_bộ', u'đường_sắt')

list = model.similar_by_word(u'計算', topn=30)
for word in list:
    print word[0], word[1]
