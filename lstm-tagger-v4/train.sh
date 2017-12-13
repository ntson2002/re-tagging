#!/bin/bash

FOLDER=./data/jcc-rre/sample/3layer

EPOCH=200
TYPE=sgd-lr_.002
TAGSCHEME=iobes
CAPDIM=0
ZERO=1
LOWER=0
WORDDIM=100
WORDLSTMDIM=100
CRF=1
ALLEMB=0
RELOAD=0
CHARDIM=0


FEATURE=pos.1.10,chunk.2.10,wh.3.10,if.4.10,s.5.10,layer1.7.10,layer2.8.10
PREEMB=./data/pretrained-emb/rre.w2v100.txt


BESTFILE=best.txt

python train.py \
	--char_dim $CHARDIM \
	--word_lstm_dim $WORDLSTMDIM \
	--train $FOLDER/train.conll \
	--dev $FOLDER/dev.conll \
	--test $FOLDER/test.conll \
	--best_outpath $BESTFILE \
	--lr_method $TYPE \
	--word_dim $WORDDIM \
	--tag_scheme $TAGSCHEME \
	--cap_dim $CAPDIM \
	--zeros $ZERO \
	--lower $LOWER \
	--reload $RELOAD \
	--external_features $FEATURE \
	--epoch $EPOCH \
	--crf $CRF \
	--pre_emb $PREEMB \
	--dropout 0.5 \
	--all_emb $ALLEMB \
	--freq_eval 1000 \
	--prefix 0
