from abc import ABC, abstractmethod

from collections import OrderedDict
import re

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
        for comment in comments:
            comment = comment.replace('<br />', '')
            tokenized_comment = tokenizer.tokenize(comment)
            for preprocessor in preprocessors:
                tokenized_comment = preprocessor.optimize(tokenized_comment)
            yield tokenized_comment


class StandardizePreprocessor(Preprocessor):
    def __init__(self):
        # \/?watch \? v = \S+
        self.regex_url = re.compile(r'[^\s|^\.]+\.[a-z]{2,3}[^\s]*')
        self.regex_number = re.compile(r'\b[0-9]+\b')
        self.regex_emoji = re.compile(r'[\S]{0,3}:[\S]{1,3}')
        self.regex_special = re.compile(r'&[a-z]+;')

    def optimize(self, tokenized_comment):
        return [self.regex_emoji.sub('EMOJII',
                                     self.regex_number.sub('NUM',
                                                           self.regex_url.sub(
                                                               'URL', self.regex_special.sub('', word))
                                                           )
                                     )
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
        return [word for word in tokenized_comment if word.lower() not in self.stop_words]


class LowercasePreprocessor(Preprocessor):
    def optimize(self, tokenized_comment):
        return [word.lower() for word in tokenized_comment]
