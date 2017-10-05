# -*- coding: utf-8 -*-
from data import SingleDataSet
from experiment import Experiment
import preprocessors

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

samples = SingleDataSet('../data/Youtube01-Psy.csv')
exp = Experiment(samples)

for preprocessors in [[preprocessors.RemoveUrls,
                       preprocessors.RemoveMultipleChars,
                       preprocessors.Lower],
                      [preprocessors.RemoveUrls,
                       preprocessors.RemoveMultipleChars],
                      [preprocessors.Lower],
                      []]:

    exp.conduct(HashingVectorizer, preprocessors)
    print("Naive Bayes: {}".format(exp.evaluate(GaussianNB(), True)))
    print("RandomForest: {}".format(exp.evaluate(
        RandomForestClassifier(n_estimators=100))))
