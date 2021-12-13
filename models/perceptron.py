from utils.decorators import debug, timer, log
import numpy as np
from models.classifier import LinearClassifier

class Perceptron(LinearClassifier):
    ''' Implementation of  Perceptrion binary classificator'''

    def __init__(self, eta=0.01, n_iter=10):
        '''Parameters:
            ---------
            eta: Learning rate eta
            
            n_iter: Number of learning iterations; epochs
        '''
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        ''' Fit data to the classifier.
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def get_params(self, **kwargs):
        return {
            'eta': self.eta,
            'n_iter': self.n_iter,
        }

    def __repr__(self):
        return f'Perceptron(eta={self.eta}, n_iter={self.n_iter})'

    def __str__(self):
        return 'Perceptron'
