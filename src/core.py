from bow import BagOfWords
from rnn import RnnClassifier
from preprocessors import *
from data import *

data = MixedFiles('data/*.csv')
preprocessors = [
    StandardizePreprocessor(),
    SlangPreprocessor('dictionaries/slang.txt'),
    StopwordPreprocessor(),
    PosLemmatizationPreprocessor(),
    StemmerPreprocessor(),
    LowercasePreprocessor()
]

x_train = [comment for comment in Preprocessor.preprocess(
    data.x_train, preprocessors)]
bag_of_words = BagOfWords(x_train).compute(min_occurrences=2)

x_train = [BagOfWords.transform(comment, bag_of_words) for comment in x_train]
x_test = [BagOfWords.transform(comment, bag_of_words)
          for comment in Preprocessor.preprocess(data.x_test, preprocessors)]

rnn = RnnClassifier(num_words=(len(bag_of_words) + 2))
rnn.fit(x_train, data.y_train)
rnn.score(x_test, data.y_test)
