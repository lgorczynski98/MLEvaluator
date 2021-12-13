import numpy as np
from models.classifier import LinearClassifier
from errors.classifier_exceptions import ClassifierWithoutRegularizationException

class LogisticRegression(LinearClassifier):

    def __init__(self, eta=0.01, n_iter=10, regularization='', c=0.1):
        ''' Parameters:
            ----------
            eta: learning rate

            n_iter: number of learning iterations

            regularization: regularization methods (available options: '', l1, l2)

            c: regularization parameter
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.regularization = regularization
        self.c = c
        if self.regularization not in ('l1', 'l2', ''):
            raise ClassifierWithoutRegularizationException(self, self.regularization)
        
        self.update_weights = getattr(self, f'calc_gradient_{self.regularization}')
        self.calc_cost = getattr(self, f'calc_cost_{self.regularization}')
        
    def _sigmoid(self, z):
        ''' Calculates sigmoid value
            Parameters:
            ----------
            z: e exponent

            Return:
            sigmoid funciton value for a given z
        '''
        return 1.0 / (1.0 + np.exp(-z))

    def calc_gradient_(self, X, errors):
        ''' Calculates gradient to update weights when without regularization
            Parameters:
            ----------
            X: features array

            errors: difference between target and predicted value
        '''
        return errors.sum(), X.T.dot(errors)

    def calc_gradient_l1(self, X, errors):
        ''' Calculates gradient to update weights when with l1 regularization
            Parameters:
            ----------
            X: features array

            errors: difference between target and predicted value
        '''
        return errors.sum() + sum(np.sign(self.w_)), self.c * X.T.dot(errors) + sum(np.sign(self.w_))

    def calc_gradient_l2(self, X, errors):
        ''' Calculates gradient to update weights when with l2 regularization
            Parameters:
            ----------
            X: features array

            errors: difference between target and predicted value
        '''
        return errors.sum() + sum(self.w_), self.c * X.T.dot(errors) + sum(self.w_)

    def calc_cost_(self, y, output):
        ''' Calculates cost function value when without regularization
            Parameters:
            ----------
            y: target value

            output: predicted value
        '''
        return np.sum(-y * np.log(output) - (1 - y) * np.log(1 - output))

    def calc_cost_l1(self, y, output):
        ''' Calculates cost function value when with l1 regularization
            Parameters:
            ----------
            y: target value

            output: predicted value
        '''
        return self.c * self.calc_cost_(y, output) + np.sum(np.absolute(self.w_))

    def calc_cost_l2(self, y, output):
        ''' Calculates cost function value when with l2 regularization
            Parameters:
            ----------
            y: target value

            output: predicted value
        '''
        return self.c * self.calc_cost_(y, output) + 0.5 * np.sum(np.power(self.w_, 2))

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
        fixed_predicted_y = np.where(predicted_y >= 0.5, 1, -1)
        for predicted, target in zip(fixed_predicted_y, y):
            if predicted != target:
                missclassifications += 1
        return missclassifications

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
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        self.errors_ = []
        N = X.shape[0]
        fixed_y = [0 if yi == -1 else 1 for yi in y]
        for _ in range(self.n_iter):
            errors = 0
            z = self.net_input(X)
            output = self._sigmoid(z)
            errors = fixed_y - output
            
            gradient_update = self.update_weights(X, errors)
            self.w_[1:] += self.eta * gradient_update[1]
            self.w_[0] += self.eta * gradient_update[0]

            self.cost_.append(self.calc_cost(y, output))
            self.errors_.append(self._calc_missclassifications(output, y))

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
        return np.where(self._sigmoid(self.net_input(X)) >= 0.5, 1, -1)

    def predict_proba(self, X):
        probas = self._sigmoid(self.net_input(X))
        probas = [[1-proba, proba] for proba in probas]
        return np.array(probas)

    def get_params(self, **kwargs):
        return {
            'eta': self.eta,
            'n_iter': self.n_iter,
            'regularization': self.regularization,
            'c': self.c,
        }

    def __repr__(self):
        return f"LogisticRegression(eta={self.eta}, n_iter={self.n_iter}, regularization='{self.regularization}', c={self.c})"

    def __str__(self):
        return 'Logistic Regression'