#! -*- coding: utf-8 -*-
import sys
import numpy as np
import gensim
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.manifold import TSNE

# Input argument as gensim model name
mymodel = sys.argv[1]
mdl_tag = gensim.models.Word2Vec.load(mymodel)

# Get word vectors (wv) * vocabulary (lexicon)
wv = mdl_tag.wv.syn0
vocabulary = mdl_tag.wv.vocab

# Set font for non-alphabet letters (ex: Korean letters)
font_fname = '/Library/Fonts/AppleGothic.ttf'	# If you are windows user, you should change the font path/name.
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

# TSNE setting
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(wv[:1000, :])

# Plot
plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()