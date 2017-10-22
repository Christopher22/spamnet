from abc import ABC
import glob
import csv

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer

class Data(ABC):
    def __init__(self, x_train, y_train, x_test, y_test, split_important=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.split_important = split_important
    
    def training_size(self):
        return len(self.x_train)
    
    def testing_size(self):
        return len(self.x_test)
    
    def find_optimum(self, pipeline, parameters):
        grid = GridSearchCV(
            pipeline, 
            n_jobs = 1, 
            verbose = 1, 
            scoring = {
                'f1': make_scorer(f1_score, pos_label=True),
                'precision': 'precision_macro',
                'recall': 'recall_macro'
            }, 
            refit = 'f1', 
            cv=(VariationGenerator(self, 3) if self.split_important else None), 
            param_grid=parameters
        )

        print("Training: {} (Training size: {}, Testing size: {})".format(
            self.__class__.__name__, 
            self.training_size(), 
            self.testing_size()
        ))
              
        grid.fit([x for x in self.x_train + self.x_test], 
                 [y for y in self.y_train + self.y_test])
        
        MSG = "{0:1.3f} (+/-{1:1.3f}; Precision: {2:1.3f}, Recall: {3:1.3f}) for {4}"
        for mean, std, prec, recall, params in sorted(
            zip(grid.cv_results_['mean_test_f1'], 
                grid.cv_results_['std_test_f1'],
                grid.cv_results_['mean_test_precision'],
                grid.cv_results_['mean_test_recall'],
                grid.cv_results_['params'],
            ), key = lambda x: x[0], reverse = True)[:3]:

            print(MSG.format(
                mean, 
                std * 2, 
                prec, 
                recall, 
                params
            ))
        
        return grid.best_params_
    
    @staticmethod    
    def _load_file(file):
         with open(file, encoding='ascii', errors='ignore') as spam:
            for row in csv.DictReader(spam, 
                                      delimiter=',', 
                                      quotechar='"', 
                                      skipinitialspace=True, 
                                      strict=True):
                yield [row['CONTENT'].strip(), row['CLASS'] == '1']
                    
class VariationGenerator:
    def __init__(self, data, n):
        self.n = n
        self.training_size = data.training_size()
        self.testing_size = data.testing_size()

    def split(self, *_):
        for _ in range(self.n):
            yield (
                shuffle(list(range(self.training_size))), 
                shuffle(list(range(self.training_size, 
                                   self.training_size + self.testing_size)))
            )
            
    def get_n_splits(self, *_):
        return self.n
        
class SingleFile(Data):
    def __init__(self, path, test_ratio=0.3):
        data = [comment for comment in Data._load_file(path)]
        x_train, x_test, y_train, y_test = train_test_split(
            [comment[0] for comment in data], 
            [comment[1] for comment in data], 
            test_size=test_ratio
        )
        super().__init__(x_train, y_train, x_test, y_test, False)
        
    
class SplittedFile(Data):
    def __init__(self, training, testing, test_ratio=0.3):
        training = [comment for comment in Data._load_file(training)]
        testing = [comment for comment in Data._load_file(testing)]
        super().__init__(
            [comment[0] for comment in training],
            [comment[1] for comment in training],
            [comment[0] for comment in testing[:int(len(training) * test_ratio)]],
            [comment[1] for comment in testing[:int(len(training) * test_ratio)]],
            True
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
        super().__init__(x_train, y_train, x_test, y_test, False)
        
class SplittedMixedFiles(Data):
    def __init__(self, paths, test_ratio=0.3):
        files = glob.glob(paths)
        shuffle(files)
        
        testing = [comment for comment in Data._load_file(files[0])]
        training = []
        for file in files[1:]:
            training.extend(comment for comment in Data._load_file(file))
            
        super().__init__(
            [comment[0] for comment in training],
            [comment[1] for comment in training],
            [comment[0] for comment in testing],
            [comment[1] for comment in testing],
            True
        )