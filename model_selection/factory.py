from abc import ABC, abstractmethod
from errors.factory_exceptions import FactoryWithoutClassifierException
from model_selection.fitters import get_fitter
from models.classifier import Classifier
import os
import pickle

class ClassifierFactory():

    def __init__(self, classifier_cls):
        self.classifier_cls = classifier_cls

    def fit_with_best_params(self, X_train, X_test, y_train, y_test, **kwargs):
        self.params = kwargs
        classifier_fitter = get_fitter(self.classifier_cls)
        self.X_train, self.X_test, self.y_train, self.y_test = classifier_fitter.prepare_data(X_train, X_test, y_train, y_test)
        best_params = classifier_fitter.get_best_params(self.X_train, self.y_train, **kwargs)
        if best_params is None or best_params == {} or best_params == [{}]:
            classifier = self.classifier_cls()
        else:
            classifier = self.classifier_cls(**best_params)
        classifier.fit(self.X_train, self.y_train)
        self.classifier = classifier
        return classifier

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def __repr__(self):
        return f'ClassifierFactory(Classifier={self.classifier_cls})'

    def save_factory(self, file_name):
        ''' Pickles the classifier into the 'classifiers' directory.
            
            Parameters:
            ----------
            file_name: Name of the pickled classifier file in the 'classifiers' directory
        '''
        dest = os.path.join('factories')
        pickle.dump(self, open(os.path.join(dest, file_name), 'wb'), protocol=4)

    @staticmethod
    def load_factory(file_name):
        ''' Loads pickled classifier from the 'classifiers' directory 

            Parameters:
            ----------
            file_name: Name of the pickled classifier file in the 'classifiers' directory

            Returns:
            Classifier loaded from 'classifiers' directory
        '''
        dest = os.path.join('factories')
        return pickle.load(open(os.path.join(dest, file_name), 'rb'))

    
def get_factory(classifier):
    return ClassifierFactory(classifier)
