from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from bow import BagOfWords
from rnn import RnnClassifier
from preprocessors import *
from data import *


def dummy(x):
    return x


# Create the necessary helper
preprocessor = PreprocessorTransformer(use_standartize=False,
                                       use_slang=False,
                                       use_stopword=False,
                                       use_lemmatization=False,
                                       use_stemmer=False,
                                       use_lowercase=False,
                                       use_punctation=False)

vectorizer = CountVectorizer(
    tokenizer=dummy, preprocessor=dummy, max_features=2000)

preprocessor_setup = {
    "pre__use_standartize": [True, False],
    "pre__use_slang": [True, False],
    "pre__use_stopword": [True, False],
    "pre__use_lemmatization": [True, False],
    "pre__use_stemmer": [True, False],
    "pre__use_lowercase": [True, False],
    "pre__use_punctation": [True, False]
}

# Create the pipelines
rnn_pipeline = Pipeline([
    ("pre", preprocessor),
    ("bow", BagOfWords(max_features=2000)),
    ("rnn", RnnClassifier())
], memory='cache')

bayes_pipeline = Pipeline([
    ("pre", preprocessor),
    ("vectorizer", vectorizer),
    ("bayes", MultinomialNB())
], memory='cache')

forest_pipeline = Pipeline([
    ("pre", preprocessor),
    ("vectorizer", vectorizer),
    ("forest", RandomForestClassifier())
], memory='cache')

# Load the datasets
datasets = [
    SingleFile('data/Youtube01-Psy.csv'),
    SplittedFile('data/Youtube01-Psy.csv', 'data/Youtube02-KatyPerry.csv'),
    MixedFiles('data/*.csv'),
    SplittedMixedFiles('data/*.csv')
]

# Start the experiment
print("Naive Bayes:")
for data in datasets:
    optimum = data.find_optimum(bayes_pipeline, {
        "bayes__alpha": [0.5, 0.75, 1.0, 1.25, 1.5],
        "bayes__fit_prior": [True, False],
    })

    print("-> Preprocessors:")
    bayes_pipeline.set_params(**optimum)
    data.find_optimum(bayes_pipeline, preprocessor_setup)

print("\nRecurrent neural network:")
for data in datasets:
    optimum = data.find_optimum(rnn_pipeline, {
        "rnn__epochs": [3, 10, 20],
        "rnn__num_hidden_neurons": [50, 100, 200],
        "rnn__dropout": [0, 0.1, 0.2],
        "rnn__rnn_type": ['gru', 'lstm', 'simple']
    })

    print("-> Preprocessors:")
    rnn_pipeline.set_params(**optimum)
    data.find_optimum(rnn_pipeline, preprocessor_setup)

print("\nRandom Forest:")
for data in datasets:
    optimum = data.find_optimum(forest_pipeline, {
        "forest__n_estimators": [10, 100, 500, 800],
        "forest__max_features": ['sqrt', 'log2', None],
    })

    print("-> Preprocessors:")
    forest_pipeline.set_params(**optimum)
    data.find_optimum(forest_pipeline, preprocessor_setup)
