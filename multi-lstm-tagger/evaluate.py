#!/usr/bin/env python

import os
import optparse
import loader
from utils import print_evaluation_result, evaluate_multilayer
from loader import prepare_dataset3
from loader import update_tag_scheme
from model_struct import Model
from tabulate import tabulate

HOME = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/journal/jpc-3layer/BI-LSTM-CRF-pretrained-emb50/06.crf-emb-feature-lr002"

default_model       = HOME + "/models/tag_scheme=iobes,char_dim=0,word_dim=50,word_bidirect=True,pre_emb=rre.w2v50.txt,all_emb=False,crf=True,dropout=0.5,external_features=pos.1.5chunk.2.5wh.3.5if.4.5s.5.5,prefix=0,model_type=multilayer"
default_test_file   = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/journal/jcc/data/train-dev-test/3layer/fold.0.test.conll"
default_json    = ""
default_txt    = ""
default_prefix = ""

default_log = False

# Read parameters from command line
optparser = optparse.OptionParser()

optparser.add_option(
    "-t", "--test_file", default=default_test_file,
    help="Test set location (conll format)"
)

optparser.add_option(
    "--out_json", default=default_json,
    help="Output location"
)

optparser.add_option(
    "--out_txt", default=default_txt,
    help="Output location"
)

optparser.add_option(
    "-d", "--detail", default=0,
    type=int,
    help="Detail"
)

optparser.add_option(
    "-m", "--model", default=default_model,
    help="Model location"
)

opts = optparser.parse_args()[0]

# Initialize model
model = Model(model_path=opts.model)
parameters = model.parameters
print "parameters: ", parameters
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


assert os.path.isfile(opts.test_file)
test_file = opts.test_file

out_txt = opts.out_txt
out_json = opts.out_json

test_sentences = loader.load_sentences(test_file, lower, zeros)
update_tag_scheme(test_sentences, tag_scheme)

test_data = prepare_dataset3(
    test_sentences, word_to_id, char_to_id, model.tag_maps, model.feature_maps, lower
)


print "input: ", test_file

from pprint import pprint
print(model.tag_maps)
pprint(model.tag_maps)

test_score, iob_test_score, result_test, _ = evaluate_multilayer(parameters, f_eval, test_sentences, test_data, model.tag_maps)

print_evaluation_result(result_test)

print "OVERALL: %f" % test_score

from pprint import pprint
pprint (result_test)
for r in result_test:
    print r["_layer"], r["result"]




if out_txt != "":
    with open(out_txt, "w") as f1:
        f1.write(tabulate([[key, parameters[key]] for key in parameters]))
        print_evaluation_result(result_test, opts.detail == 1, f1)
        print "file", out_txt, "has been created !"

if out_json != "":
    import json
    with open(out_json, "w") as f2:
        json.dump(result_test,  f2, indent=4)
        print "file", out_json, "has been created !"


