# Recurrent neural network-based models for recognizing requisite and effectuation parts in legal texts


Requirements:

*  Python 2.7, with Numpy and Theano installed.


Two implemented models:

* [lstm-tagger-v4](https://github.com/ntson2002/re-tagging/tree/master/lstm-tagger-v4): Implementation of single BI-LSTM-CRF with additional features to recognize non-overlapping RE parts by modeling the RRE task as the single layer sequence labeling task (1 layer). 

* [multi-lstm-tagger](https://github.com/ntson2002/re-tagging/tree/master/multi-lstm-tagger): Implementation of Multilayer BiLSTM-CRF model, and Multilayer BiLSTM-MLP-CRF to recognize overlapping RE parts by modeling the RRE task as the multilayer sequence labeling task (n layer).

Reference:


Nguyen, Truong-Son, Le-Minh Nguyen, Satoshi Tojo, Ken Satoh, and Akira Shimazu. “Recurrent Neural Network-Based Models for Recognizing Requisite and Effectuation Parts in Legal Texts.” Artificial Intelligence and Law 26, no. 2 (June 2018): 169–199. [https://doi.org/10.1007/s10506-018-9225-1](https://doi.org/10.1007/s10506-018-9225-1).

