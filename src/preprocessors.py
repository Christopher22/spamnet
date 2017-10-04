""" A module for handling dynamic chained preprocessors. """

from abc import ABC, abstractmethod
from itertools import groupby


class Preprocessor(ABC):
    """ A chainable preprocessor. """

    @abstractmethod
    def __iter__(self):
        """ Returns a generator for the actual preprocessing. """
        raise NotImplementedError

    @staticmethod
    def optimize(tweets, preprocessors):
        """ Optimize a tweet out of a iterator by applying a chain of preprocessors on it. """
        chain = None
        for i, preprocessor in enumerate(preprocessors):
            chain = (globals()[preprocessor])(tweets if i == 0 else chain)
        for tweet in chain:
            yield tweet


class RemoveUrls(Preprocessor):
    """ Replace links by a fixed keyword. """

    def __init__(self, tweets):
        """ Creates a new preprocessor. """
        self.tweets = tweets

    def __iter__(self):
        for tweet in self.tweets:
            tweet.text = [x if not x.startswith(
                'http') else 'URL' for x in tweet.text]
            yield tweet


class RemoveMultipleChars:
    "Remove multiple, equal characters ('HIIII!!!!' -> 'HI!')"

    def __init__(self, tweets):
        """ Creates a new preprocessor. """
        self.tweets = tweets

    def __iter__(self):
        for tweet in self.tweets:
            tweet.text = [''.join(ch for ch, _ in groupby(word))
                          for word in tweet.text]
            yield tweet
