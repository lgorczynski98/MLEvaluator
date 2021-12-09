from models.decision_tree import DecisionTreeClassifier
from models.classifier import Classifier
import random
import numpy as np
from collections import Counter
from errors.classifier_exceptions import ClassifierWithoutMaxFeatureMethodException

class RandomForrestClassifier(Classifier):

    class RandomForrestTreeClassifier(DecisionTreeClassifier):
        ''' Decision tree with a method to calculate maximum number of random features
            that are taken into consideration when splitting the data
        '''

        def __init__(self, criterion='gini', max_depth=None, max_features='sqrt'):
            ''' Parameters:
                ----------
                criterion: impurity calculation method(available options: gini, entropy, classification_error)

                max_depth: maximum tree depth

                max_features: method to calculate maximum number of random features 
                    that are taken into consideraiton when splitting the data
                    (available options: sqrt, log2) 
            '''
            self.max_features = max_features
            if not self.max_features in ('sqrt', 'log2'):
                raise ClassifierWithoutMaxFeatureMethodException(self, self.max_features)
            super().__init__(criterion, max_depth)

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
            if self.max_features == 'sqrt':
                self.max_number_of_features = int(np.floor(np.sqrt(len(X[0]))))
            elif self.max_features == 'log2':
                self.max_number_of_features = int(np.floor(np.log2(len(X[0]))))
            super().fit(X, y)

        def find_best_split(self, X, y):
            ''' Find data split that gives the greatest gain value 
                and question that provides it
                Parameters:
                ----------
                X: array-like of shape (n_samples, n_features)
                        Training vectors, where `n_samples` is the number of samples and
                        `n_features` is the number of predictors.

                y : array-like of shape (n_samples,) or (n_samples, n_targets)
                        Target vectors, where `n_samples` is the number of samples and
                        `n_targets` is the number of response variables.

                Returns:
                Best gain and question that provides it
            '''
            best_gain = 0
            best_question = None
            curr_uncertainty = self.impurity(y)
            n_features = len(X[0])
            features = random.sample(range(n_features), self.max_number_of_features)
            for feature in features:
                values = self.unique_values_with_biggest_margin(X, feature)
                for val in values:
                    question = super().Question(feature, val, super())
                    true_x, true_y, false_x, false_y = self.partition(X, y, question)
                    if len(true_y) == 0 or len(false_y) == 0:
                        continue
                    gain = self.info_gain(true_y, false_y, curr_uncertainty)
                    if(gain >= best_gain):
                        best_gain, best_question = gain, question
            return best_gain, best_question
            
        def __repr__(self):
            return f'RandomForrestTreeClassifier(criterion={self.criterion}, max_depth={self.max_depth}, max_features={self.max_features})'
    

    def __init__(self, n_estimators, criterion='gini', max_depth=None, max_features='sqrt'):
        ''' Parameters:
            ----------
            n_estimators: number of tree in the random forrest classificator

            criterion: decision trees' criterion for impurity calculation

            max_depth: maximum depth of the trees

            max_features: method to calculate maximum number of random features 
                    that are taken into consideraiton when splitting the data
                    (available options: sqrt, log2)
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.estimators = [self.RandomForrestTreeClassifier(criterion, max_depth, max_features) for _ in range(self.n_estimators)]

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
        for estimator in self.estimators:
            estimator.fit(X, y)

    def mode(self, labels):
        ''' Calculates most common label in the given dataset

            Parameters:
            ----------
            labels: array containing labels

            Returns:
            Most common label
        '''
        return Counter(labels).most_common(1)[0][0]

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
            predictions = []
            for estimator in self.estimators:
                predictions.append(estimator.single_predict(x))
            labels.append(self.mode(predictions))
        return np.array(labels)

    def predict_proba(self, X):
        proba = []
        for x in X:
            estimator_proba = []
            for estimator in self.estimators:
                estimator_proba.append(estimator.single_predict_proba(x))
            estimator_proba = np.array(estimator_proba)
            estimator_proba_sum = np.sum(estimator_proba, axis=0)
            proba.append([float(estimator_proba_sum[0] / self.n_estimators), float(estimator_proba_sum[1] / self.n_estimators)])
        return np.array(proba)

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
        predicted_labels = self.predict(X)
        score = 0
        for predicted_y, yi, in zip(predicted_labels, y):
            if predicted_y == yi:
                score += 1
        return score / len(y)

    def print_forrest(self, spacing=''):
        ''' Simple method to visualize all the trees in the forrest
            Parameters:
            ----------
            spacing: stretches branch display
        '''
        for idx, estimator in enumerate(self.estimators):
            print(f'Estimator: {idx+1}')
            estimator.print_tree(spacing)

    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
        }

    def __repr__(self):
        return f'RandomForredtClassifier(n_estimators={self.n_estimators}, criterion={self.criterion}, max_depth={self.max_depth}, max_features={self.max_features})'

    def __str__(self):
        return 'Random Forrest'