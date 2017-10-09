from collections import Counter


class BagOfWords:
    def __init__(self, data=[]):
        self.bow = Counter()
        for sample in data:
            self.add(sample)

    def add(self, data):
        self.bow.update(data)

    def compute(self, min_occurrences=1, max_features=None):
        output = {}
        counter = 2
        for word, occurences in self.bow.most_common(max_features):
            if occurences >= min_occurrences:
                output[word] = counter
                counter += 1
        return output

    @staticmethod
    def transform(tokenized_text, bag_of_words):
        return [bag_of_words[word] if word in bag_of_words else 1 for word in tokenized_text]
