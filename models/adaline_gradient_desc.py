import numpy as np
from models.classifier import LinearClassifier

class Adaline(LinearClassifier):
    ''' Implementation of adaptive linear gradient classificator '''

    def __init__(self, eta=0.01, n_iter=10):
        '''Parameters:
        -------------
            eta: Learning rate eta

            n_iter: Number of learning iterations; epochs
        '''
        self.eta = eta
        self.n_iter = n_iter

    def _calc_missclassifications(self, predicted_y, y):
        ''' Calculate number of missclassifications during learning iteration; epoch

            Parameters:
            ----------
            predicted_y: array-like of shape (n_samples,) or (n_samples, n_targets)
                    predicted class labels

            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Return:
            int: Number of missclassified predictions
        '''
        missclassifications = 0
        for predicted, target in zip(predicted_y, y):
            if abs(predicted - target) > 1:
                missclassifications += 1
        return missclassifications

    def fit(self, X, y):
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
        self.cost_ = []
        self.errors_ = []
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            self.errors_.append(self._calc_missclassifications(output, y))
            
        return self

    def get_params(self, **kwargs):
        return {
            'eta': self.eta,
            'n_iter': self.n_iter,
        }

    def __repr__(self):
        return f'Adaline(eta={self.eta}, n_iter={self.n_iter})'

    def __str__(self):
        return 'Adaline'