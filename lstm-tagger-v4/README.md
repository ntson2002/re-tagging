# lstm-tagger-v4: BI-LSTM-CRF with features for single layer sequence labeling task


## Initial setup

To use the tagger, you need Python 2.7, with Numpy and Theano installed.


## Train a model which recognize RE parts in *1 layer*

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set. The following script will train the model to recognize RE parts for layer 3 which uses word, syntactic features and tag of layer 1,2 as features (see parameter `--external_features`):

```
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
PREFIX=layer3

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
	--all_emb 1 \
	--freq_eval 1000 \
	--prefix=$PREFIX > logs/log.txt

```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```

### Reproduce the result
```
np.random.seed(1234)
```
### Japanese Civil Code RRE corpus


List of columns (start from 0):

* 0: Head word
* 1: POS
* 2: NP chunks
* 3: WH clause
* 4: Clause begin with preposition
* 5: Clause in S node of the syntactic tree
* 6: *Unused*
* 7: Gold tag at layer 1 (Requisite and effectuation parts)
* 8: Gold tag at layer 2 (Requisite and effectuation parts)
* 9: Gold tag at layer 3 (Unless part)

Feature description format `FEATURE=name1.column_index1.embedding_size1;...`. For example: 

```
	FEATURE=pos.1.10,chunk.2.10,wh.3.10,if.4.10,s.5.10,layer1.7.10,layer2.8.10
```

Data format:

```
Actions	NNS	B-NP	O	O	O	O	B-R	B-E	O
for	IN	I-NP	O	O	O	O	I-R	O	O
preservation	NN	I-NP	O	O	O	O	I-R	O	O
of	IN	I-NP	O	O	O	O	I-R	O	O
possession	NN	E-NP	O	O	O	O	I-R	O	O
may	MD	O	O	O	O	O	O	I-E	O
be	VB	O	O	O	O	O	O	I-E	O
brought	VBN	O	O	O	O	O	O	I-E	O
so	RB	O	O	O	O	O	O	I-E	O
long	RB	O	O	O	O	O	O	I-E	O
as	IN	O	O	B-IF	O	O	O	I-E	O
the	DT	B-NP	O	I-IF	B-S	O	O	I-E	O
danger	NN	I-NP	O	I-IF	I-S	O	O	I-E	O
of	IN	I-NP	O	I-IF	I-S	O	O	I-E	O
disturbance	NN	E-NP	O	I-IF	I-S	O	O	I-E	O
exists	VBZ	E-VP	O	I-IF	E-S	O	O	I-E	O
.	.	O	O	E-IF	O	O	O	O	O
									
In	IN	O	O	O	O	O	B-R	O	O
such	JJ	B-NP	O	O	O	O	I-R	O	O
cases	NNS	E-NP	O	O	O	O	I-R	O	O
,	,	O	O	O	O	O	O	O	O
the	DT	B-NP	O	O	O	O	B-E	O	O
proviso	NN	I-NP	O	O	O	O	I-E	O	O
to	TO	I-NP	O	O	O	O	I-E	O	O
the	DT	I-NP	O	O	O	O	I-E	O	O
preceding	VBG	I-NP	O	O	O	O	I-E	O	O
paragraph	NN	E-NP	O	O	O	O	I-E	O	O
shall	MD	O	O	O	O	O	I-E	O	O
apply	VB	O	O	O	O	O	I-E	O	O
mutatis	JJ	B-NP	O	O	O	O	I-E	O	O
mutandis	NNS	E-NP	O	O	O	O	I-E	O	O
if	IN	O	O	B-IF	O	O	B-R	O	O
possessed	NNP	B-NP	O	I-IF	O	O	I-R	O	O
Thing	NNP	E-NP	O	I-IF	O	O	I-R	O	O
is	VBZ	O	O	I-IF	O	O	I-R	O	O
likely	JJ	O	O	I-IF	O	O	I-R	O	O
to	TO	O	O	I-IF	O	O	I-R	O	O
be	VB	O	O	I-IF	O	O	I-R	O	O
damaged	VBN	O	O	I-IF	O	O	I-R	O	O
by	IN	O	O	I-IF	O	O	I-R	O	O
construction	NN	E-NP	O	E-IF	O	O	I-R	O	O
.	.	O	O	O	O	O	O	O	O
									
Any	DT	B-NP	O	O	O	O	B-R	B-E	O
person	NN	E-NP	O	O	O	O	I-R	I-E	O
who	WP	O	B-WHNP	O	O	O	I-R	O	O
is	VBZ	O	I-WHNP	O	O	O	I-R	O	O
neither	RB	O	I-WHNP	O	O	O	I-R	O	O
an	DT	B-NP	I-WHNP	O	O	O	I-R	O	O
incorporated	JJ	I-NP	I-WHNP	O	O	O	I-R	O	O
association	NN	I-NP	I-WHNP	O	O	O	I-R	O	O
nor	CC	I-NP	I-WHNP	O	O	O	I-R	O	O
an	DT	I-NP	I-WHNP	O	O	O	I-R	O	O
incorporated	JJ	I-NP	I-WHNP	O	O	O	I-R	O	O
foundation	NN	E-NP	E-WHNP	O	O	O	I-R	O	O
shall	MD	O	O	O	O	O	O	I-E	O
not	RB	O	O	O	O	O	O	I-E	O
use	VB	O	O	O	O	O	O	I-E	O
in	IN	O	O	O	O	O	O	I-E	O
its	PRP$	B-NP	O	O	O	O	O	I-E	O
name	NN	E-NP	O	O	O	O	O	I-E	O
the	DT	B-NP	O	O	O	O	O	I-E	O
words	NNS	E-NP	O	O	O	O	O	I-E	O
``	``	O	O	O	O	O	O	I-E	O
incorporated	VBN	O	O	O	O	O	O	I-E	O
association	NN	B-NP	O	O	O	O	O	I-E	O
''	''	I-NP	O	O	O	O	O	I-E	O
or	CC	I-NP	O	O	O	O	O	I-E	O
``	``	I-NP	O	O	O	O	O	I-E	O
incorporated	JJ	I-NP	O	O	O	O	O	I-E	O
foundation	NN	I-NP	O	O	O	O	O	I-E	O
''	''	E-NP	O	O	O	O	O	I-E	O
,	,	O	O	O	O	O	O	I-E	O
or	CC	O	O	O	O	O	O	I-E	O
other	JJ	B-NP	O	O	O	O	O	I-E	O
words	NNS	E-NP	O	O	O	O	O	I-E	O
which	WDT	O	B-WHNP	O	O	O	O	I-E	O
is	VBZ	O	I-WHNP	O	O	O	O	I-E	O
likely	JJ	O	I-WHNP	O	O	O	O	I-E	O
to	TO	O	I-WHNP	O	O	O	O	I-E	O
be	VB	O	I-WHNP	O	O	O	O	I-E	O
mistaken	VBN	O	I-WHNP	O	O	O	O	I-E	O
for	IN	O	I-WHNP	O	O	O	O	I-E	O
those	DT	B-NP	I-WHNP	O	O	O	O	I-E	O
words	NNS	E-NP	E-WHNP	O	O	O	O	I-E	O
.	.	O	O	O	O	O	O	O	O
									

```
