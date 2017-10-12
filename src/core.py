from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from bow import BagOfWords
from rnn import RnnClassifier
from preprocessors import *
from data import *

data = MixedFiles('data/*.csv')

# Find the best RNN parameter
rnn_pipeline = Pipeline([
    ("pre", PreprocessorTransformer()),
    ("bow", BagOfWords()),
    ("rnn", RnnClassifier())
], memory='cache')

data.find_optimum(rnn_pipeline, {
    "rnn__epochs": [3, 10, 20],
    "rnn__num_hidden_neurons": [50, 100, 200],
    "rnn__dropout": [0, 0.1, 0.2],
    "rnn__rnn_type": ['gru', 'lstm', 'simple']
})

# Find the best RandomForest parameter


def dummy(x):
    return x


forest_pipeline = Pipeline([
    ("pre", PreprocessorTransformer()),
    ("vectorizer", CountVectorizer(tokenizer=dummy, preprocessor=dummy)),
    ("forest", RandomForestClassifier(n_estimators=500))
], memory='cache')

data.find_optimum(forest_pipeline, {
    "forest__n_estimators": [10, 100, 500, 800],
    "forest__max_features": ['sqrt', 'log2', None],
})
