from abc import ABC, abstractmethod
import numpy as np
import os
import pickle

class Classifier(ABC):
    
    @abstractmethod
    def fit(self, X, y):
        ''' Fit data to the classifier.
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.
        '''

    @abstractmethod
    def predict(self, X):
        ''' Predicts the class of the input data (binary classification)
            
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

            Return:
            int (1- or 1): Class label
        '''

    @abstractmethod
    def score(self, X, Y):
        '''Calculates the prediction accuracy on the given dataset.
        
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Return:
            float (0-1): Ration of correctly predicted data -> correct_predictions/all_predictions
        '''  

    @abstractmethod
    def get_params(self, **kwargs):
        '''Returns model parameters'''

    def get_effective_params(self):
        return self.get_params()

    def save_classifier(self, file_name):
        ''' Pickles the classifier into the 'classifiers' directory.
            
            Parameters:
            ----------
            file_name: Name of the pickled classifier file in the 'classifiers' directory
        '''
        dest = os.path.join('classifiers')
        pickle.dump(self, open(os.path.join(dest, file_name), 'wb'), protocol=4)

    @staticmethod
    def load_classifier(file_name):
        ''' Loads pickled classifier from the 'classifiers' directory 

            Parameters:
            ----------
            file_name: Name of the pickled classifier file in the 'classifiers' directory

            Returns:
            Classifier loaded from 'classifiers' directory
        '''
        dest = os.path.join('classifiers')
        return pickle.load(open(os.path.join(dest, file_name), 'rb'))



class LinearClassifier(Classifier, ABC):
    '''Base class of implemented classificators'''

    def net_input(self, X):
        ''' Calculates the dot product of the weights and input data 

            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

            Return:
            float: Activation of the function
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        ''' Predicts the class of the input data (binary classification)
            
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

            Return:
            int (1- or 1): Class label
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def score(self, X, Y):
        '''Calculates the prediction accuracy on the given dataset.
        
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Return:
            float (0-1): Ration of correctly predicted data -> correct_predictions/all_predictions
        '''  
        score = 0
        for xi, target in zip(X, Y):
            if self.predict(xi) == target:
                score += 1
        return score/len(X)

