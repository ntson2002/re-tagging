## Multilayer BiLSTM-CRF and Multilayer BiLSTM-MLP-CRF

### Initial setup

To use the tagger, you need Python 2.7, with Numpy and Theano installed.

### Input data

Format:
* The input data follows the CoNLL format. 
* Each line represents a word and its features. 
* Sentences are split using a blank line. 

Below is an example of training data for JCC-RRE task which has 9 columns:
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
									

```

### Training 

#### Parameters:
* `--model_type`: 
	* `multilayer` : Multi-BiLSTM-CRF
	* `struct` : Multi-BiLSTM-MLP-CRF with 1 dense layer in MLPs 
	* `struct_mlp`: Multi-BiLSTM-MLP-CRF with 2 dense layer in MLPs 

* `--tag_columns_string` : Indexs of gold columns seperated by commas. E.g `7,8,9`
* `--external_features` : Features description. The format is follows: `name1.column1.size1,name2.column2.size2`

#### A sample tranining script:
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

```
PROGRAM=<PROGRAM_FOLDER>
TEST=<CONNL_FILE>
MODEL=<saved_model_path>
OUTPUT=<OUTPATH>
python $PROGRAM/predict_file.py -m $MODEL -t $TEST --out_file $OUTPUT 
```

### Sample output 

