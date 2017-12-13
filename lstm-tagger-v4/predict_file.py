#!/usr/bin/env python

import os
import codecs
import optparse
import loader
from utils import models_path, predict, eval_temp, eval_path
from loader import prepare_dataset2
from loader import update_tag_scheme
from model import Model

# default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/06emb-crf-f12345-lr002/layer0/models/prefix=0"
# default_test_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/new/test-1layer/fold.0.test.conll"
# default_out_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/jp-rre/jpc-new-v1/01jpc-emb-crf-d100/06emb-crf-f12345-lr002/layer0/fold.0.test.conll.txt"

default_model = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/models/vn_p30c30_layer1"
default_test_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/data/testb1_sample_notag.conll"
default_out_file = "/home/s1520203/Bitbucket/lstm-crf-tagging/lstm-tagger-v4/data/testb1_sample_predicted_tag.conll"

default_prefix = ""

default_log = False

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-t", "--test_file", default=default_test_file,
    help="Test set location (conll format)"
)

optparser.add_option(
    "-o", "--out_file", default=default_out_file,
    help="Output location"
)

optparser.add_option(
    "-m", "--model", default=default_model,
    help="Model location"
)

optparser.add_option(
    "-a", "--add_o_tag", default=False,
    help="Add a column O at the end"
)

opts = optparser.parse_args()[0]

# Check parameters validity
assert os.path.isfile(opts.test_file)

if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(model_path=opts.model)
parameters = model.parameters

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = model.parameters['tag_scheme']

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

print  model.id_to_tag
print 'Reloading previous model...'
_, f_eval = model.build(training=False, **parameters)
model.reload()

from tabulate import tabulate
embeddings_params, model_params = model.count_parameters()
print tabulate(embeddings_params, ["name", "shape", "size"])
print tabulate(model_params, ["name", "shape", "size"])


print "Parameters: \n  *", "\n  * ".join([str(k) + " = " + str(v) for k, v in model.parameters.items()])
# print "Parameters: \n", model.parameters

test_file = opts.test_file
out_file = opts.out_file

test_sentences = loader.load_sentences(test_file, lower, zeros)
update_tag_scheme(test_sentences, tag_scheme)

test_data = prepare_dataset2(
    test_sentences, word_to_id, char_to_id, tag_to_id, model.feature_maps, lower
)

print "input: ", test_file, ":", len(test_sentences), len(test_data)
print "output: ", out_file

import datetime
t1 = datetime.datetime.now()
predict(parameters, f_eval, test_sentences, test_data, model.id_to_tag, out_file, add_O_tags=opts.add_o_tag)
t2 = datetime.datetime.now()
print "done in ", (t2-t1).total_seconds(), "seconds"


print "finish !!!"
