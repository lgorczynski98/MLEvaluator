import numpy as np
from models.classifier import LinearClassifier
from errors.classifier_exceptions import ClassifierWithoutKernelException

class SupportVectorClassifier(LinearClassifier):

    def __init__(self, eta, n_iter, c, kernel=None, d=2, r=1, gamma=-2, k1=1, k2=1):
        ''' Parameters:
            ----------
            eta: learning rate

            n_iter: number of learning iterations

            c: parameter c value

            kernel: kernel basis function (available values: None, 
                    polynomial, kernel_basis, tangent_neural_network)

            d: polynomial kernel exponent value

            r: polynomial kernel shift value

            gamma: radial basis kernel gamma parameter value

            k1: tangent neural netword k1 parameter value

            k2: tangent neural netword k2 parameter value
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.c = c
        self.kernel = kernel
        self.d = d
        self.r = r
        self.gamma = gamma
        self.k1 = k1
        self.k2 = k2
        if self.kernel is not None and not hasattr(self, f'{self.kernel}_kernel'):
            raise ClassifierWithoutKernelException(self, self.kernel)
        if self.kernel is not None:
            self.calc_kernel = getattr(self, f'{kernel}_kernel')

    def update_weights(self, xi, yi):
        ''' Update weight based on one sample
            Parameters:
            ----------
            xi: array - one sample data

            yi: sample's label
        '''
        distance = yi * np.dot(xi, self.w_[1:] + self.w_[0])
        if distance >= 1:
            self.w_[1:] -= self.eta * self.w_[1:]
        else:
            self.w_[1:] -= self.eta * (self.w_[1:] - self.c * np.dot(xi, yi))
            self.w_[0] -= self.eta * yi
        
        # distance = max(0, 1 - yi * (np.dot(xi, self.w_[1:]) + self.w_[0]))
        # if distance > 0:
        #     self.w_[1:] -= self.eta * (self.c * self.w_[1:] + yi * xi)
        # else:
        #     self.w_[1:] -= self.eta * self.c * self.w_[1:]
        # self.w_[0] -= self.eta * (yi - np.dot(self.w_[1:], xi))

        # if yi * (np.dot(xi, self.w_[1:]) + self.w_[0]) <= 1:
        #     self.w_[1:] = (1 - self.eta) * self.w_[1:] + self.eta * self.c * yi * xi
        #     self.w_[0] = (1 - self.eta) * self.w_[0] + self.eta * self.c * yi
        # else:
        #     self.w_ = (1 - self.eta) * self.w_

    def polynomial_kernel(self, x1, x2):
        ''' Polynomial kernel transformation
            Parameters:
            ----------
            x1: feature matrix

            x2: feature matrix 
        '''
        return (np.dot(x1, x2.T) + self.r) ** self.d

    def radial_basis_kernel(self, x1, x2):
        ''' Radial basis kernel transformation
            Parameters:
            ----------
            x1: feature matrix

            x2: feature matrix 
        '''
        return np.exp(self.gamma * np.square(x1[:, np.newaxis] - x2).sum(axis=2))

    def tangent_neural_network_kernel(self, x1, x2):
        ''' Tangent neural network kernel transformation
            Parameters:
            ----------
            x1: feature matrix

            x2: feature matrix 
        '''
        return np.tanh(self.k1 * np.dot(x1, x2.T) + self.k2)

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
        self.X = X
        if self.kernel is None:
            self.w_ = np.zeros(1 + X.shape[1])
        else:
            gram_matrix = self.calc_kernel(X, X)
            self.w_ = np.zeros(1 + gram_matrix.shape[1])
        self.cost_ = []
        self.erors_ = []
        for _ in range(self.n_iter):
            if self.kernel is None:
                for xi, yi in zip(X, y):
                    self.update_weights(xi, yi)
            else:
                for xi, yi in zip(gram_matrix, y):
                    self.update_weights(xi, yi)
        return self

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
        if self.kernel is None:
            return super().predict(X)
        else: 
            Xk = self.calc_kernel(X, self.X)
            return super().predict(Xk)

    def score(self, X, y):
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
        if self.kernel is None:
            Xk = X
        else:
            Xk = self.calc_kernel(X, self.X)
        score = 0
        for xi, target in zip(Xk, y):
            if super().predict(xi) == target:
                score += 1
        return score/len(X)

    def get_params(self):
        return {
            'eta': self.eta,
            'n_iter': self.n_iter,
            'c': self.c,
            'kernel': self.kernel,
            'd': self.d,
            'r': self.r,
            'gamma': self.gamma,
            'k1': self.k1,
            'k2': self.k2,
        }

    def __repr__(self):
        return (f'SupportVectorClassifier(eta={self.eta}, n_iter={self.n_iter}, c={self.c}, '
            f'kernel={self.kernel}, d={self.d}, r={self.r}, gamma={self.gamma}, k1={self.k1}, k2={self.k2})')

    def __str__(self):
        return 'Support Vector Classifier'