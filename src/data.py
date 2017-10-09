from abc import ABC
import glob
import random

from sklearn.model_selection import train_test_split


class Data(ABC):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def training(self):
        return (self.x_train, self.y_train)

    def testing(self):
        return (self.x_test, self.y_test)

    @staticmethod
    def _load_file(file):
        with open(file, encoding='ascii', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue

                txt = line.strip().split(',')
                if txt[3].startswith('"'):
                    txt = ''.join(txt[3:])
                    yield [txt[1: -2].strip(), txt[-1] == '1']
                else:
                    yield [txt[3].strip(), txt[4] == '1']


class SingleFile(Data):
    def __init__(self, path, test_ratio=0.3):
        data = [comment for comment in Data._load_file(path)]
        x_train, x_test, y_train, y_test = train_test_split(
            [comment[0] for comment in data],
            [comment[1] for comment in data],
            test_size=test_ratio
        )
        super().__init__(x_train, y_train, x_test, y_test)


class SplittedFile(Data):
    def __init__(self, training, testing, test_ratio=0.3):
        training = [comment for comment in Data._load_file(training)]
        testing = [comment for comment in Data._load_file(testing)]
        super().__init__(
            [comment[0] for comment in training],
            [comment[1] for comment in training],
            [comment[0]
                for comment in testing[:int(len(training) * test_ratio)]],
            [comment[1]
                for comment in testing[:int(len(training) * test_ratio)]]
        )


class MixedFiles(Data):
    def __init__(self, paths, test_ratio=0.3):
        data = []
        for file in glob.iglob(paths):
            data.extend(comment for comment in Data._load_file(file))
        x_train, x_test, y_train, y_test = train_test_split(
            [comment[0] for comment in data],
            [comment[1] for comment in data],
            test_size=test_ratio
        )
        super().__init__(x_train, y_train, x_test, y_test)


class SplittedMixedFiles(Data):
    def __init__(self, paths, test_ratio=0.3):
        files = glob.glob(paths)
        random.shuffle(files)

        testing = [comment for comment in Data._load_file(paths[0])]
        training = []
        for i, file in enumerate(files):
            if i == 0:
                continue
            training.extend(comment for comment in Data._load_file(file))

        super().__init__(
            [comment[0] for comment in training],
            [comment[1] for comment in training],
            [comment[0] for comment in testing],
            [comment[1] for comment in testing]
        )
