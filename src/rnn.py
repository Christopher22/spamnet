from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences


class RnnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, input_length=32, num_words=5000, embedding_dimension=32, batch_size=32, epochs=3, num_hidden_neurons=100):
        self.input_length = input_length
        self.num_words = num_words
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_hidden_neurons = num_hidden_neurons

        self._rnn = None

    def fit(self, X, y=None):
        assert (y is not None), "Y is required"
        X = pad_sequences(X, self.input_length)

        self._rnn = Sequential()
        self._rnn.add(Embedding(
            self.num_words, self.embedding_dimension, input_length=self.input_length))
        self._rnn.add(GRU(self.num_hidden_neurons))
        self._rnn.add(Dense(1))
        self._rnn.add(Activation('sigmoid'))
        self._rnn.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
        self._rnn.fit(X, y, epochs=self.epochs,
                      batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X, y=None):
        if self._rnn is None:
            raise RuntimeError("Fitting required before prediction!")

        X = pad_sequences(X, self.input_length)
        return [prob[0] >= 0.5 for prob in self._rnn.predict(X, batch_size=self.batch_size)]

    def score(self, X, y=None):
        assert (y is not None), "Y is required"

        prediction = self.predict(X)
        return f1_score(y, prediction)
