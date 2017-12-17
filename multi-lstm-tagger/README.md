## Multilayer BI-LSTM-CRF

### Initial setup

To use the tagger, you need Python 2.7, with Numpy and Theano installed.

### Input data

The input data follows the CoNLL format. Below is an example of training data for JCC-RRE task:

The data has 9 columns:

* Column 0:
* Column 1, 2, 3, 4, 5: Syntactic features 
* Column 6: Unused 
* Column 7, 8, 9: Gold labels for 3 layers 

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

### Training 
```
FOLDER=<DATA_FOLDER>
PROGRAM=<PROGRAM_FOLDER>

EPOCH=200
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
GOLDCOLUMNS=7,8,9

CRF=1
ALLEMB=0
PREEMB=$MY_HOME/Bitbucket/lstm-crf-tagging/experiments/jp-rre/data/pretrained/rre.w2v50.txt

START=0
END=9

MODEL_TYPE=struct_mlp
for (( FOLD=$START; FOLD<=$END; FOLD++ ))
do
	BESTFILE=best.$FOLD.txt
	python -u $PROGRAM/train_struct.py \
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
		--model_type=$MODEL_TYPE \
		--all_emb $ALLEMB > logs/log.$FOLD.$HOSTNAME.txt  2>&1 &
done
```

### Predict 


### Sample output 

