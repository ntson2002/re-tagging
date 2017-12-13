#!/usr/bin/env bash

MODEL_PATH=$PWD/rre-sample-scripts/models/jpl-rre-f36

#cd $HOME/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/
cd ../../../lstm-tagger-v4
export PYTHONPATH=$PWD:$PYTHONPATH
python ./apps/jpl-rre/rre-api.py --save $MODEL_PATH --port 8126