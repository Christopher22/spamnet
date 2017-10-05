''' A module for the efficient testing of different vectorizers, preprocessors and algorithms. '''

from preprocessors import Preprocessor
from data import DataSet

from sklearn.metrics import f1_score


class Experiment:
    ''' An efficient experiment for testing different vectorizers, preprocessors and algorithm. '''

    def __init__(self, data):
        ''' Create a new experiment. '''

        if not isinstance(data, DataSet):
            raise ValueError("Expected a DataSet as input.")

        self.data = data
        self._training = None
        self._testing = None
        self._bow = None
        self._training_results = [y.is_spam for y in data.training]
        self._testing_results = [y.is_spam for y in data.test]

    def conduct(self, vectorizer, preprocessors):
        ''' Conducts an experiment with a vectorizer and preprocessors. '''

        if self._training is not None:
            self.reset()

        if preprocessors:
            self._training = [
                tweet for tweet in Preprocessor.optimize(self.data.training, preprocessors)]
            self._testing = [
                tweet for tweet in Preprocessor.optimize(self.data.test, preprocessors)]
        else:
            self._training = self.data.training
            self._testing = self.data.test

        self._bow = vectorizer(
            (tweet.text for tweet in self._training + self._testing))
        self._training = self._bow.transform(
            (' '.join(x.text) for x in self._training))
        self._testing = self._bow.transform(
            (' '.join(x.text) for x in self._testing))

    def reset(self):
        ''' Undo an experiment conduction. '''

        self._training = None
        self._testing = None
        self._bow = None

    def evaluate(self, classifier, expect_numpy=False):
        ''' Evaluates a given classifier with the f - score of its prediction on the test set. '''

        if self._training is None:
            raise ValueError("Experiment was not conducted!")

        classifier.fit(
            self._training if not expect_numpy else self._training.toarray(),
            self._training_results
        )

        return f1_score(
            self._testing_results,
            classifier.predict(
                self._testing if not expect_numpy else self._testing.toarray())
        )
