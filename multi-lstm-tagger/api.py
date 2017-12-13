import codecs
import json
import os

import numpy as np
import web

import loader
from loader import prepare_dataset3
from loader import update_tag_scheme_multilayer
from model_struct import Model
from utils import predict_multilayer

urls = (
        # '/api/tagging/text=(.*)', 'api_do_tagging',
        '/api/tagging', 'api_do_tagging',
        '/demo/(.*)', 'demo')

# loading the model
# model_path = os.getenv("HOME") + "/Bitbucket/lstm-crf-tagging/experiments/vn-ner-2layer/run-multi-lstm/models/tag_scheme=iobes,char_dim=25,word_dim=100,word_bidirect=True,pre_emb=train.txt.w2vec100,all_emb=False,crf=True,dropout=0.5,external_features=pos.1.30chunk.2.30,prefix=p30c30"


# Initialize model
def load_model(model_path):
    model = Model(model_path=model_path)
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
    return [f_eval, model, parameters, lower, zeros, tag_scheme, word_to_id, char_to_id]




def extract_re_from_conll_data(conll_data):
    def iob_tag(tag):
        return tag[0]

    def logic_tag(tag):
        return "O" if tag == "O" else tag[2:]

    def get_re_parts(tags, words):
        # print tags
        # print words
        results = []
        i = 0
        while (i < len(tags)):
            while (i < len(tags)) and iob_tag(tags[i]) == "O":
                i = i + 1
            if i < len(tags):
                semantic_tag = logic_tag(tags[i])
                word_list = [words[i]]
                i = i + 1
                while i < len(tags) and iob_tag(tags[i]) in {"O", "I"}:
                    if iob_tag(tags[i]) == "I":
                        word_list.append(words[i])
                    i = i + 1
                results.append({"tag": semantic_tag, "text": " ".join(word_list)})
        return results

    # from pprint import pprint
    # pprint(conll_data)
    nlayer = 3
    results = []
    for data in conll_data:
        words = [x[0] for x in data]

        re_list = []
        for layer in range(1, nlayer + 1):
            tags = [x[layer] for x in data]
            re_list.append(get_re_parts(tags, words))

        results.append({"text": " ".join(words), "re": re_list})

    return results


class demo:
    def GET(self, q):
        import codecs
        web.header('Content-Type', 'text/html')
        with codecs.open("demo.html", "r", "utf-8") as f:
            html_text = f.read()
        return html_text

class api_do_tagging:
    def POST(self):


        post_data = web.input(_method='post')

        feature_type = post_data["feature"]
        f_eval, model, parameters, lower, zeros, tag_scheme, word_to_id, char_to_id = web.pre_load_data[feature_type]

        if post_data.has_key("format"):
            format = post_data["format"]
        else:
            format = "json"

        # print post_data


        # read the data from request and save into file
        file_id = np.random.randint(1000000, 2000000)

        input_path = os.path.join("temp", "user.%i.input" % file_id)
        with codecs.open(input_path, "w", "utf-8") as f:
            f.write(post_data["text"])

        gold_colums = [int(x['column']) for x in model.tag_maps]
        test_sentences = loader.load_sentences(input_path, lower, zeros)
        update_tag_scheme_multilayer(test_sentences, gold_colums, tag_scheme)

        test_data = prepare_dataset3(
            test_sentences, word_to_id, char_to_id, model.tag_maps, model.feature_maps, lower
        )

        # print test_data[0]

        out_sentences = predict_multilayer(parameters, f_eval, test_sentences, test_data, model.tag_maps, None)

        results = extract_re_from_conll_data(out_sentences)

        if format == "json":
            data = {"conll_data": out_sentences, "text_data": results}
            web.header('Content-Type', 'application/json')
            return json.dumps(data, indent=4, sort_keys=True, encoding="utf-8")
        else:
            conll_text = "\n\n".join(["\n".join(["\t".join(l) for l in item]) for item in out_sentences])
            return conll_text


class TaggerAPIApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))


if __name__ == "__main__":
    # model_path = os.getenv("HOME") + "/Bitbucket/lstm-crf-tagging/experiments/journal/jpc-3layer/07.struct-mlp-BI-LSTM-CRF-pretrained-emb50/00.crf-emb-nofeature-lr002/models/prefix=7,model_type=struct_mlp"
    # folder = os.getenv("HOME") + "/Bitbucket/lstm-crf-tagging/experiments/journal/jpc-3layer/07.struct-mlp-BI-LSTM-CRF-pretrained-emb50"
    folder = os.getenv("HOME") + "/Bitbucket/lstm-crf-tagging/experiments/journal/jpc-3layer/03.BI-LSTM-CRF-pretrained-emb50"
    model_paths = {
        "word"          : folder + "/00.crf-emb-nofeature-lr002/models/prefix=1,model_type=multilayer",
        "word+syntactic": folder + "/06.crf-emb-feature-lr002/models/prefix=1,model_type=multilayer"
    }

    models_data = {}
    for key, path in model_paths.items():
        f_eval, model, parameters, lower, zeros, tag_scheme, word_to_id, char_to_id = load_model(path)
        models_data[key] = [f_eval, model, parameters, lower, zeros, tag_scheme, word_to_id, char_to_id]



    web.pre_load_data = models_data

    app = TaggerAPIApplication(urls, globals())
    app.run(port=8124)


