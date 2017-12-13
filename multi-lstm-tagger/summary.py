#!/usr/bin/env python

"""
[{'_layer': 0,
  'confusion_table': 'ID\tNE\tTotal\tI-R\tI-E\tO\tB-R\tE-R\tE-E\tB-E\tS-R\tPredict\tCorrect\tRecall\tPrecision\tF1\t\n0\tI-R\t3373\t2948\t103\t93\t7\t212\t9\t1\t0\t3173\t2948\t87.400\t92.909\t90.070\t\n1\tI-E\t1963\t160\t1508\t146\t6\t11\t130\t1\t1\t1666\t1508\t76.821\t90.516\t83.108\t\n2\tO\t1305\t50\t47\t1191\t7\t5\t4\t1\t0\t1448\t1191\t91.264\t82.251\t86.524\t\n3\tB-R\t242\t9\t7\t13\t208\t0\t0\t3\t2\t233\t208\t85.950\t89.270\t87.579\t\n4\tE-R\t0\t0\t0\t0\t0\t0\t0\t0\t0\t228\t0\t0.000\t0.000\t0.000\t\n5\tE-E\t0\t0\t0\t0\t0\t0\t0\t0\t0\t143\t0\t0.000\t0.000\t0.000\t\n6\tB-E\t148\t6\t1\t5\t5\t0\t0\t131\t0\t137\t131\t88.514\t95.620\t91.930\t\n7\tS-R\t0\t0\t0\t0\t0\t0\t0\t0\t0\t3\t0\t0.000\t0.000\t0.000\t\n5986/7031 (85.13725%)',
  'conlleval': u'accuracy:  90.03%; precision:  84.86%; recall:  82.28%; FB1:  83.55\n                E: precision:  85.42%; recall:  80.92%; FB1:  83.11  144\n                R: precision:  84.52%; recall:  83.13%; FB1:  83.82  239',
  'fb1': 83.55,
  'result': {'FB1': 83.547557840617,
             'accuracy': 90.0298677286304,
             'correct': 325,
             'corrected_token': 5986,
             'precision': 84.8563968668407,
             'processed_tokens': 7031,
             'recall': 82.2784810126582,
             'total_found': 383,
             'total_phrased': 395,
             'total_token': 7031}},
 {'_layer': 1,
  'confusion_table': 'ID\tNE\tTotal\tO\tI-E\tE-E\tB-E\tS-E\tI-R\tB-R\tE-R\tPredict\tCorrect\tRecall\tPrecision\tF1\t\n0\tO\t6109\t5907\t183\t12\t6\t1\t0\t0\t0\t6005\t5907\t96.693\t98.368\t97.524\t\n1\tI-E\t862\t90\t720\t51\t1\t0\t0\t0\t0\t903\t720\t83.527\t79.734\t81.586\t\n2\tE-E\t0\t0\t0\t0\t0\t0\t0\t0\t0\t63\t0\t0.000\t0.000\t0.000\t\n3\tB-E\t60\t8\t0\t0\t50\t2\t0\t0\t0\t57\t50\t83.333\t87.719\t85.470\t\n4\tS-E\t0\t0\t0\t0\t0\t0\t0\t0\t0\t3\t0\t0.000\t0.000\t0.000\t\n5\tI-R\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0.000\t0.000\t0.000\t\n6\tB-R\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0.000\t0.000\t0.000\t\n7\tE-R\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0.000\t0.000\t0.000\t\n6677/7031 (94.96515%)',
  'conlleval': u'accuracy:  95.72%; precision:  78.57%; recall:  79.38%; FB1:  78.97\n                E: precision:  78.57%; recall:  79.38%; FB1:  78.97  98',
  'fb1': 78.97,
  'result': {'FB1': 78.974358974359,
             'accuracy': 95.7189588963163,
             'correct': 77,
             'corrected_token': 6677,
             'precision': 78.5714285714286,
             'processed_tokens': 7031,
             'recall': 79.3814432989691,
             'total_found': 98,
             'total_phrased': 97,
             'total_token': 7031}},
 {'_layer': 2,
  'confusion_table': 'ID\tNE\tTotal\tO\tI-U\tB-U\tE-U\tPredict\tCorrect\tRecall\tPrecision\tF1\t\n0\tO\t6191\t6191\t0\t0\t0\t6418\t6191\t100.000\t96.463\t98.200\t\n1\tI-U\t792\t222\t569\t1\t0\t569\t569\t71.843\t100.000\t83.615\t\n2\tB-U\t24\t2\t0\t22\t0\t23\t22\t91.667\t95.652\t93.617\t\n3\tE-U\t24\t3\t0\t0\t21\t21\t21\t87.500\t100.000\t93.333\t\n6803/7031 (96.75722%)',
  'conlleval': u'accuracy:  96.76%; precision:  91.30%; recall:  87.50%; FB1:  89.36\n                U: precision:  91.30%; recall:  87.50%; FB1:  89.36  23',
  'fb1': 89.36,
  'result': {'FB1': 89.3617021276596,
             'accuracy': 96.757218034419,
             'correct': 21,
             'corrected_token': 6803,
             'precision': 91.304347826087,
             'processed_tokens': 7031,
             'recall': 87.5,
             'total_found': 23,
             'total_phrased': 24,
             'total_token': 7031}}]
"""

import json

result_folder = "/home/s1520203/Bitbucket/lstm-crf-tagging/experiments/journal/jpc-3layer/BI-LSTM-CRF-pretrained-emb50/06.crf-emb-feature-lr002/temp"
exp_name = "06.crf-emb-feature-lr002"
template = "eval.$FOLD.json"
start = 0
end = 9
headers = ['folder_layer', 'layer' 'test_file', 'total_ntokens', 'total_phrases', 'total_found', 'total_correct', 't_precision', 't_recall', 't_F1']
table = []
for i in range(start, end+1):
    name = template.replace("$FOLD", str(i))
    path = result_folder + "/" + name
    with open(path, "r") as f:
        data = json.load(f)
        for r in data:
            layer = "layer" + str(r["_layer"])
            fold = str(i)
            ntokens = r["result"]["total_token"]
            nphrases = r["result"]["total_phrased"]
            nfound = r["result"]["total_found"]
            ncorrect = r["result"]["correct"]
            precision = r["result"]["precision"]
            recall = r["result"]["recall"]
            F1 = r["result"]["FB1"]

            table.append([exp_name + ":" + layer, layer, name, ntokens, nphrases, nfound, ncorrect, precision, recall, F1])

print '****DETAILS****'
print '\t'.join(headers)
print '\n'.join('\t'.join([str(x) for x in items]) for items in table)
print

table2 = []
for l in [['layer0'], ['layer1'], ['layer2'], ['layer0', 'layer1'], ['layer0', 'layer1', 'layer2']]:
    total_phrases = 0
    total_found = 0
    total_correct = 0
    total_ntokens = 0
    for r in table:
        folder_layer, layer, test_file, ntokens, nphrases, nfound, ncorrect, precision, recall, F1 = r

        if layer in l:
            total_ntokens += int(ntokens)
            total_phrases += int(nphrases)
            total_found += int(nfound)
            total_correct += int(ncorrect)

    t_precision = int(total_correct) / (1.0 * int(total_found))
    t_recall = int(total_correct) / (1.0 * int(total_phrases))
    t_F1 = 2 * t_precision * t_recall / (t_precision + t_recall)

    ls = '-'.join(l)

    table2.append([exp_name + ':' + ls, total_ntokens, total_phrases, total_found, total_correct, t_precision, t_recall, t_F1])

headers2 = ['folder_layer', 'total_ntokens', 'total_phrases', 'total_found', 'total_correct', 't_precision', 't_recall', 't_F1']
print '****OVERALL****'
print '\n'.join('\t'.join([str(x) for x in items]) for items in table2)
print
