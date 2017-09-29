""" A module for efficient handling of the different datasets. """

from datetime import datetime
from abc import ABC, abstractmethod
import random


class Data:
    """ A single comment in the dataset. """

    def __init__(self, author, text, date, isSpam):
        self.author = author
        self.text = text.split()
        self.date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
        self.is_spam = isSpam

    @staticmethod
    def parse(line):
        """ Parses a line to a data object. """
        data = line.strip().split(',')
        if len(data) == 5:
            try:
                data = Data(data[1], data[3], data[2], (data[4] == '1'))
            except ValueError:
                data = None
        else:
            data = None
        return data


class DataSet(ABC):
    """ A abstract dataset providing both training and testing set. """

    @property
    @abstractmethod
    def training(self):
        """ Returns the training set. """
        raise NotImplementedError

    @property
    @abstractmethod
    def test(self):
        """ Returns the testing set. """
        raise NotImplementedError


class SingleDataSet(DataSet):
    """ A dataset for training and testing on a single file. """

    def __init__(self, path):
        data = []
        with open(path, 'r') as file:
            data = [data for data in [Data.parse(
                line) for line in file.readlines()] if data is not None]

        self._training = data[:int(len(data) * 0.7)]
        self._test = data[:int(len(data) * 0.7)]

    @property
    def training(self):
        return self._training

    @property
    def test(self):
        return self._test


class SplittedSingleDataSet(DataSet):
    """ A dataset for training on one file and testing on another. """

    def __init__(self, path_training, path_testing):
        training = SingleDataSet(path_training)
        testing = SingleDataSet(path_testing)

        self._training = training.training + training.test
        self._test = testing.training[:int(len(self._training) * 0.3)]

    @property
    def training(self):
        return self._training

    @property
    def test(self):
        return self._test


class CombinedDataSet(DataSet):
    """ A dataset for training and testing on multiple files. """

    def __init__(self, paths):
        self._training = []
        self._test = []

        for path in paths:
            tmp = SingleDataSet(path)
            self._training.extend(tmp.training)
            self._test.extend(tmp.test)

        random.shuffle(self._training)
        random.shuffle(self._test)

    @property
    def training(self):
        return self._training

    @property
    def test(self):
        return self._test


class MultiDataSet(DataSet):
    """ A dataset for training on multiple files and testing on another file. """

    def __init__(self, paths):
        train = CombinedDataSet(paths[:len(paths) - 1])
        test = SingleDataSet(paths[-1])

        self._training = train.training + train.test
        self._test = test.training + test.test

    @property
    def training(self):
        return self._training

    @property
    def test(self):
        return self._test
