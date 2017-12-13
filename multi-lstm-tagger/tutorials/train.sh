#!/usr/bin/env bash


FOLDER=$MY_HOME/Bitbucket/lstm-crf-tagging/experiments/journal/jcc/data/train-dev-test/2layer
PROGRAM=$MY_HOME/Bitbucket/lstm-crf-tagging/struct-lstm

EPOCH=300
TYPE=sgd-lr_.002
TAGSCHEME=iobes
CAPDIM=0
ZERO=1
LOWER=0
WORDDIM=50
WORDLSTMDIM=50

RELOAD=0
CHARDIM=0
FEATURE=pos.1.5,chunk.2.5,wh.3.5,if.4.5,s.5.5
GOLDCOLUMNS=7,8

CRF=1
ALLEMB=1
PREEMB=$MY_HOME/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v50.txt


for FOLD in {0..9}
do
	BESTFILE=best.$FOLD.txt
	python $PROGRAM/train.py \
		--char_dim $CHARDIM \
		--word_lstm_dim $WORDLSTMDIM \
		--train $FOLDER/fold.$FOLD.train.conll \
		--dev $FOLDER/fold.$FOLD.dev.conll \
		--test $FOLDER/fold.$FOLD.test.conll \
		--best_outpath $BESTFILE \
		--reload $RELOAD \
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
		--prefix=$FOLD \
		--tag_columns_string $GOLDCOLUMNS \
		--all_emb $ALLEMB
done


