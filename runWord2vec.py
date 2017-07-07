# -*- coding: utf-8 -*-
# koword2vec.py

import sys
import gensim
import logging
import multiprocessing


# Model configuration for training
config = {
    'min_count': 1,  # Ignore a word if it is observed less than 'min_count' times
    'size': 300,  # Number of hidden neurons; dimension of word embeddings
    'sg': 1,  # 0: CBOW, 1: skip-gram
    'hs': 1, # 0: None, 1: apply hierarchical softmax
    'negative': 0, # negative sampling
    'batch_words': 10000, # Number of words in a batch
    'iter': 10,  # iteration
    'window': 2, # window size (how many nearby words to be considered as context words)
    'workers': multiprocessing.cpu_count() # If an error occurs, switch this to small integers
}

corpus_fname = sys.argv[1]
model_name = sys.argv[2]
# -----------------------------------------------------------------------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# -----------------------------------------------------------------------------------------
print('[1] Begin reading sentences')
sentences_vocab = gensim.models.word2vec.LineSentence(corpus_fname)
sentences_train = gensim.models.word2vec.LineSentence(corpus_fname)
print('==> Corpus reader set-up completed!')
# -----------------------------------------------------------------------------------------
print('[2] Set up a model for word2vec training')
model = gensim.models.Word2Vec(**config)
print('==> Model set-up completed!')
# -----------------------------------------------------------------------------------------
print('[3] Begin building vocabulary')
model.build_vocab(sentences_vocab)
print('==> Vocabulary building completed!')
# -----------------------------------------------------------------------------------------
print('[4] Begin training word vectors')
try:
	model.train(sentences_train)
except ValueError:	# For newer version of Gensim
	model.train(sentences_train, total_examples = model.corpus_count, epochs = model.iter)
print('==> Training of word embedding completed!')
# -----------------------------------------------------------------------------------------
print('Now saving model...')
model.save(model_name)
print('==> Successfully saved!')
# -----------------------------------------------------------------------------------------
