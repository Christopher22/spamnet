from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


class BagOfWords(BaseEstimator, TransformerMixin):
    def __init__(self, min_occurrences=2, max_features=None):
        self.counter = Counter()
        self.min_occurrences = min_occurrences
        self.max_features = max_features
        self.bow = None

    def fit(self, X, y=None):
        for x in X:
            self.counter.update(x)

        self.bow = {}
        i = 2
        for word, occurences in self.counter.most_common(self.max_features):
            if occurences >= self.min_occurrences:
                self.bow[word] = i
                i += 1

    def transform(self, X, y=None):
        if self.bow is None:
            raise RuntimeError("Fitting required before transform!")

        output = []
        for x in X:
            output.append(
                [self.bow[word] if word in self.bow else 1 for word in x])
        return output

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def size(self):
        return len(self.bow)
