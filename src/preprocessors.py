""" A module for handling dynamic chained preprocessors. """

from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """ A chainable preprocessor. """

    def __init__(self, tweets):
        """ Creates a new preprocessor. """
        self.tweet = tweets

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
