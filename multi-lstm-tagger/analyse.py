#!/usr/bin/env python

import optparse
import os

import loader
from loader import prepare_dataset3
from loader import update_tag_scheme
from model_struct import Model
from utils import predict_multilayer


folder = "/home/s1520203//Bitbucket/lstm-crf-tagging/experiments/journal/jpc-3layer/03.BI-LSTM-CRF-pretrained-emb50/06.crf-emb-feature-lr002"
default_model = folder + "/models/prefix=2,model_type=multilayer"


default_log = False

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-m", "--model", default=default_model,
    help="Model location"
)


opts = optparser.parse_args()[0]

# Initialize model
model = Model(model_path=opts.model)
parameters = model.parameters

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = model.parameters['tag_scheme']

# Load reverse mappings
word_to_id, char_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char]
]

print 'Reloading previous model...'
_, f_eval = model.build(training=False, **parameters)
model.reload()

import numpy as np
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

print model.components['transitions0'].get_value().shape
t = model.components['transitions0'].get_value()

for i in range(t.shape[0]):
    # tt = normalize(t[i])
    tt = t[i]
    print "\t".join([str(x) for x in tt])

for tm in model.tag_maps:
    tags = [tm['id_to_tag'][i] for i in range(len(tm['id_to_tag']))]
    print "\t".join(tags)

    # print tm['id_to_tag']

print model.tag_maps




