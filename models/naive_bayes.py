import numpy as np
from models.classifier import Classifier

class GausianNaiveBayesClassifier(Classifier):

    def fit(self, X, y):
        ''' Fit data to the classifier. Calculates means and stds for every feature
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.
        '''
        positive_samples, negative_samples = [], []
        for xi, yi in zip(X, y):
            if yi == 1:
                positive_samples.append(xi)
            else:
                negative_samples.append(xi)

        self.positive_label_proba = float(len(positive_samples) / len(y))
        self.negative_label_proba = float(len(negative_samples) / len(y))
        positive_samples = np.array(positive_samples)
        negative_samples = np.array(negative_samples)
        self.positive_mean_std = [(np.mean(sample), np.std(sample)) for sample in positive_samples.T]
        self.negative_mean_std = [(np.mean(sample), np.std(sample)) for sample in negative_samples.T]
        return self

    def calculate_proba(self, x, mean, std):
        ''' Calculates likelihood for a feature value using feature's mean and std
            Calculation based on a Gaussian normal distribution

            Parameters:
            ----------
            x: value of a one of data samples' features

            mean: mean feature value

            std: standard deriviation of a feature

            Returns:
            float
        '''
        return (1 / np.sqrt(2 * np.pi * np.power(std, 2))) * np.exp((-np.power(x - mean, 2)) / (2 * np.power(std, 2)))

    def predict_single_proba(self, x):
        positive_proba = (self.positive_label_proba * 
                            np.prod([self.calculate_proba(xi, self.positive_mean_std[idx][0], self.positive_mean_std[idx][1]) for idx, xi in enumerate(x)]))
        negative_proba = (self.negative_label_proba * 
                            np.prod([self.calculate_proba(xi, self.negative_mean_std[idx][0], self.negative_mean_std[idx][1]) for idx, xi in enumerate(x)]))
        return [negative_proba, positive_proba]

    def predict_proba(self, X):
        proba = []
        for x in X:
            proba.append(self.predict_single_proba(x))
        return np.array(proba)

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
        labels = []
        for x in X:
            negative_proba, positive_proba = self.predict_single_proba(x)
            if positive_proba >= negative_proba:
                labels.append(1)
            else:
                labels.append(-1)
        return np.array(labels)

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
        y_predicted = self.predict(X)
        for y, y_pred in zip(Y, y_predicted):
            if y == y_pred:
                score += 1
        return score / len(Y)

    def get_params(self, **kwargs):
        return {}

    def __repr__(self):
        return 'GausianNaiveBayes()'

    def __str__(self):
        return 'Gausian Naive Bayes'