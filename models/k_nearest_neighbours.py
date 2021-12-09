import numpy as np
from errors.classifier_exceptions import ClassifierWithoutMetricException
from collections import Counter
from models.classifier import Classifier

class KNearestNeighbours(Classifier):

    def __init__(self, k, metric = 'euclidean'):
        ''' Parameters:
            -----------
            k: determines how many neighbours are used to predict label

            metric: metric used to calculate distance betweend samples
                (euclidean)
        '''
        self.k = k
        self.metric = metric
        if not hasattr(self, metric):
            raise ClassifierWithoutMetricException(self, metric)
        self.distance = getattr(self, metric)

    def euclidean(self, point1, point2):
        ''' Calculates distance between points using euclidean metric
            
            Parameters:
            ----------
            point1: Sample data

            point2: Sample data

            Returns:
            Euclidean distance betweend given samples
        '''
        distance = 0
        for xi, xj in zip(np.nditer(point1), np.nditer(point2)):
            distance += np.power(xi - xj, 2)
        return np.sqrt(distance)

    def manhattan(self, point1, point2):
        ''' Calculates distance between points using manhattan metric
            
            Parameters:
            ----------
            point1: Sample data

            point2: Sample data

            Returns:
            Euclidean distance betweend given samples
        '''
        distance = 0
        for xi, xj in zip(np.nditer(point1), np.nditer(point2)):
            distance += np.absolute(xi - xj)
        return distance

    def fit(self, X_train, y_train):
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
        self.X_train = X_train
        self.y_train = y_train

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
            distances = [(self.distance(x, train_sample), label) for train_sample, label in zip(self.X_train, self.y_train)]
            sorted_distances = sorted(distances)
            k_nearest_distances = sorted_distances[:self.k]
            k_nearest_labels = [label for distance, label in k_nearest_distances]
            labels.append(self.mode(k_nearest_labels))
        return np.array(labels)

    def predict_proba(self, X):
        proba = []
        for x in X:
            distances = [(self.distance(x, train_sample), label) for train_sample, label in zip(self.X_train, self.y_train)]
            sorted_distances = sorted(distances)
            k_nearest_distances = sorted_distances[:self.k]
            k_nearest_labels = [label for distance, label in k_nearest_distances]
            proba.append([float(k_nearest_labels.count(-1) / self.k), float(k_nearest_labels.count(1) / self.k)])
        return np.array(proba)
            
    def mode(self, labels):
        ''' Calculates most common label in the given dataset

            Parameters:
            ----------
            labels: array containing labels

            Returns:
            Most common label
        '''
        return Counter(labels).most_common(1)[0][0]

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
            if(y == y_pred):
                score += 1
        return score / len(Y)

    def get_params(self):
        return {
            'k': self.k,
            'metric': self.metric,
        }

    def __repr__(self):
        return f'KNearestNeighbours(k={self.k}, metric={self.metric})'

    def __str__(self):
        return 'K Nearest Neighbours'