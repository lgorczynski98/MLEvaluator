from model_selection.factory import get_factory, ClassifierFactory
from sklearn.model_selection import train_test_split
from evaluators.evaluator import Evaluator
import logging
import time
import os
import pickle
import time
import tracemalloc
from model_selection.usage import Usage


class ModelSelector(object):

    def __init__(self, *args):#*args - list of **kwargs
        self.classifiers_cls = []
        self.params = []
        for kwargs in args:
            self.classifiers_cls.append(kwargs.pop('classifier'))
            self.params.append(kwargs)

        self.factories = [get_factory(classifier_cls) for classifier_cls in self.classifiers_cls]
        self.evaluator = None
        self.usage = {factory: Usage() for factory in self.factories}
    
    def fit(self, X, y, test_size=0.25, save_factories=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        for factory, params in zip(self.factories, self.params):
            self.fit_and_measure_usage(factory, X_train, X_test, y_train, y_test, params)
            if save_factories:
                self.save_factory(factory)
        self.evaluator = Evaluator(*self.factories)
        self.evaluator.add_usages(self.usage)
        return self

    def fit_and_measure_usage(self, factory, X_train, X_test, y_train, y_test, params):
        tracemalloc.start()
        
        start_time = time.perf_counter()

        self.fit_single_classifier(factory, X_train, X_test, y_train, y_test, params)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        size, peak = tracemalloc.get_traced_memory()
        memory_usage = peak - size
        tracemalloc.stop()

        self.usage[factory].add_time(run_time)
        self.usage[factory].add_memory(memory_usage)

    def fit_single_classifier(self, factory, X_train, X_test, y_train, y_test, params):
        logging.warning(f'Starting fitting with best params for {factory.classifier_cls}')
        classifier = factory.fit_with_best_params(X_train, X_test, y_train, y_test, **params)
        logging.warning(f'Best params: {repr(classifier)}')

    def save_factory(self, factory, file_name=None):
        if file_name is None:
            timestamp = int(time.time())
            file_name = f'{str(factory.classifier)}_{timestamp}.pkl'
            file_name = file_name.replace(' ', '_').lower()
        factory.save_factory(file_name)

    def save_factories(self):
        for factory in self.factories:
            self.save_factory(factory)

    def add_factory(self, factory):
        self.classifiers_cls.append(factory.classifier_cls)
        self.params.append(factory.params)
        self.factories.append(factory)
        self.evaluator = Evaluator(*self.factories)

    def save(self, file_name):
        logging.info(f'Saving model selector in classifiers/{file_name}')
        dest = os.path.join('classifiers')
        pickle.dump(self, open(os.path.join(dest, file_name), 'wb'), protocol=4)

    @staticmethod
    def load(file_name):
        logging.info(f'Loading model selector from classifiers/{file_name}')
        dest = os.path.join('classifiers')
        return pickle.load(open(os.path.join(dest, file_name), 'rb'))