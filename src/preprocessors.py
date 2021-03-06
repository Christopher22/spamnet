from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from collections import OrderedDict
import re
import inspect

from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import wordnet, stopwords
from nltk.stem.porter import PorterStemmer


class Preprocessor(ABC):
    @abstractmethod
    def optimize(self, tokenized_comment):
        raise NotImplementedError()

    @staticmethod
    def preprocess(comments, preprocessors):
        tokenizer = TweetTokenizer()
        html_cleaner = re.compile('<.+?>')
        for comment in comments:
            comment = html_cleaner.sub('', comment)
            tokenized_comment = tokenizer.tokenize(comment)
            for preprocessor in preprocessors:
                tokenized_comment = preprocessor.optimize(tokenized_comment)
            yield tokenized_comment


class StandardizePreprocessor(Preprocessor):
    def __init__(self):
        self.regex_url = re.compile(r'[^\s|^\.]+\.[a-z]{2,3}[^\s]*')
        self.regex_number = re.compile(r'\b[0-9]+\b')
        self.regex_emoji = re.compile(r'[\S]{0,3}:[\S]{1,3}')
        self.regex_special = re.compile(r'&[a-z]+;')

    def optimize(self, tokenized_comment):
        return [self.regex_emoji.sub('EMOJII',
                                     self.regex_number.sub('NUM',
                                                           self.regex_url.sub('URL',
                                                                              self.regex_special.sub('', word))))
                for word in tokenized_comment]


class SlangPreprocessor(Preprocessor):
    def __init__(self, normalisation_dictionary):
        self.double_character = re.compile(r'(.)\1{2,}')

        self.dictionary = {}
        with open(normalisation_dictionary, encoding='ascii', errors='ignore') as f:
            for line in f:
                key, value = line.strip().split("\t")
                self.dictionary[key] = value

    def optimize(self, tokenized_comment):
        output = []
        for word in tokenized_comment:
            word = self.double_character.sub(r'\1\1', word)
            if word.lower() in self.dictionary:
                word = self.dictionary[word.lower()]
            output.append(word)
        output = list(OrderedDict.fromkeys(output))
        return output


class PosLemmatizationPreprocessor(Preprocessor):
    def __init__(self):
        self.regex_non_word = re.compile(r"[^a-zA-Z\.!?']")
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _tag_to_wordnet(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def optimize(self, tokenized_comment):
        output = []
        for word in tokenized_comment:
            word = self.regex_non_word.sub('', word).strip()
            if len(word) > 0:
                output.append(word)

        for i, (word, tag) in enumerate(pos_tag(output)):
            pos_type = self._tag_to_wordnet(tag)
            if pos_type is not None:
                output[i] = self.lemmatizer.lemmatize(word, pos=pos_type)
            else:
                output[i] = word

        return output


class StemmerPreprocessor(Preprocessor):
    def __init__(self):
        self.porter = PorterStemmer()

    def optimize(self, tokenized_comment):
        return [self.porter.stem(word) for word in tokenized_comment]


class StopwordPreprocessor(Preprocessor):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def optimize(self, tokenized_comment):
        return [word for word in tokenized_comment
                if word.lower() not in self.stop_words]


class LowercasePreprocessor(Preprocessor):
    def optimize(self, tokenized_comment):
        return [word.lower() for word in tokenized_comment]


class PunctationRemover(Preprocessor):
    def __init__(self):
        self.char_only = re.compile(r'[^a-zA-Z]')

    def optimize(self, tokenized_comment):
        output = []
        for word in tokenized_comment:
            word = self.char_only.sub('', word)
            if len(word) > 0:
                output.append(word)
        return output


class PreprocessorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 use_standartize=True,
                 use_slang=True,
                 use_stopword=True,
                 use_lemmatization=True,
                 use_stemmer=True,
                 use_lowercase=True,
                 use_punctation=True):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        preprocessors = []
        if self.use_standartize:
            preprocessors.append(StandardizePreprocessor())
        if self.use_slang:
            preprocessors.append(SlangPreprocessor('dictionaries/slang.txt'))
        if self.use_stopword:
            preprocessors.append(StopwordPreprocessor())
        if self.use_lemmatization:
            preprocessors.append(PosLemmatizationPreprocessor())
        if self.use_stemmer:
            preprocessors.append(StemmerPreprocessor())
        if self.use_lowercase:
            preprocessors.append(LowercasePreprocessor())
        if self.use_punctation:
            preprocessors.append(PunctationRemover())
        return [tokenized for tokenized in Preprocessor.preprocess(X, preprocessors)]

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
