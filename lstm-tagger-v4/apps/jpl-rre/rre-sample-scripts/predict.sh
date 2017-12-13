#!/usr/bin/env bash

PROGRAM=$MY_HOME/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/predict_file.py
MODEL=$MY_HOME/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/apps/jpl-rre/rre-sample-scripts/models/jpl-rre-f36
#TEST=sample-sentence.conll
TEST=rre_jpl_pp2.conll
OUT=$1

python $PROGRAM --test_file $TEST --out_file $OUT --model $MODEL

#python predict_file.py --test_file rre-sample-scripts/jpl-sample.conll --out_file rre-sample-scripts/jpl-sample.out.conll --model models/jpl_rre_f36
