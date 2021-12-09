import numpy as np
import operator
from errors.classifier_exceptions import ClassifierWithoutCriterionException
from models.classifier import Classifier


class DecisionTreeClassifier(Classifier):

    class Question:
        ''' Question that splist the data in two new branches
        '''
        def __init__(self, feature, value, decision_tree):
            ''' Parameters:
                ----------
                feature: feature column number that values were used to split data

                value: feature value

                decision_tree: references to decision tree
            '''
            self.feature = feature
            self.value = value
            self.is_numeric = decision_tree.is_numeric

        def match(self, sample):
            ''' Check if sample data matches the quesiton
                Parameters:
                ----------
                sample: sample's feature data

                Returns:
                Boolean value
            '''
            val = sample[self.feature]
            if self.is_numeric(val):
                return val >= self.value
            else:
                return val == self.value

        def __repr__(self):
            condition = '>=' if self.is_numeric(self.value) else '=='
            return f'Is X[][{self.feature}] {condition} {self.value} ?'


    class Leaf:
        ''' Leaf node of a tree
        '''
        def __init__(self, predictions):
            ''' Parameters:
                ----------
                predictions: array with labels and number of it's occurances in the leaf
            '''
            self.predictions = predictions


    class DecisionNode:
        ''' Decision node that poits to its children and a question that splited the data
        '''
        def __init__(self, question, true_branch, false_branch):
            ''' Parameters:
                ----------
                question: question that splited the data

                true_branch: child node with data that match the question

                false_branch: child node with data that don't match the question
            '''
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch


    def __init__(self, criterion='gini', max_depth=None):
        ''' Parameters:
            ----------
            criterion: impurity calculation method(available options: gini, entropy, classification_error)

            max_depth: maximum tree depth
        '''
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        if not hasattr(self, f'{criterion}_impurity'):
            raise ClassifierWithoutCriterionException(self, criterion)
        self.impurity = getattr(self, f'{criterion}_impurity')

    def class_counts(self, y):
        ''' Counts number of label occurances
            Parameters:
            ----------
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Returns:
            array with labels and number of its occurances
        '''
        label_counts = {}
        for yi in y:
            if yi not in label_counts:
                label_counts[yi] = 0
            label_counts[yi] += 1
        return label_counts

    def is_numeric(self, value):
        ''' Checks if a value is an int or float instance
            Parameters:
            value: value to be checked

            Returns:
            Boolean value 
        '''
        return isinstance(value, int) or isinstance(value, float)

    def unique_values(self, X, feature):
        ''' Finds unique values in a feature column
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.
            
            feature: number of the feature column that is searched

            Returns:
            set of values in the given feature column
        '''
        return set([x[feature] for x in X])

    def unique_values_with_biggest_margin(self, X, feature):
        ''' Finds unique values in a feature column, providing the biggest possible margin between samples
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.
            
            feature: number of the feature column that is searched

            Returns:
            set of values in the given feature column
        '''
        vals = list(self.unique_values(X, feature))
        if not self.is_numeric(vals[0]):
            return set(vals)
        else:
            vals.sort()
            middle_vals = []
            for i in range(len(vals) - 1):
                middle_vals.append(vals[i] + (vals[i + 1] - vals[i]) / 2)
            return set(middle_vals)

    def partition(self, X, y, question):
        ''' Partitions data into 4 arrays with a given question
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.
            question: question to split data

            Returns:
            true_x: array with samples' features that match the question

            true_y array with samples' labels that match the question
            
            false_x: array with samples' features that don't match the question
            
            false_y array with samples' labels that don't match the question
        '''
        true_x, true_y, false_x, false_y = [], [], [], []
        for xi, yi in zip(X, y):
            if question.match(xi):
                true_x.append(xi)
                true_y.append(yi)
            else: 
                false_x.append(xi)
                false_y.append(yi)
        return true_x, true_y, false_x, false_y

    def gini_impurity(self, y):
        ''' Calculates the impurity using the gini method
            Parameters:
            ----------
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Returns:
            float <0;1>
        '''
        counts = self.class_counts(y)
        impurity = 1
        for label in counts:
            probability_of_label = counts[label] / float(len(y))
            impurity -= probability_of_label ** 2
        return impurity

    def entropy_impurity(self, y):
        ''' Calculates the impurity using the entropy method
            Parameters:
            ----------
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Returns:
            float <0;1>
        '''
        counts = self.class_counts(y)
        impurity = 0
        for label in counts:
            probability_of_label = counts[label] / float(len(y))
            impurity += probability_of_label * np.log2(probability_of_label)
        return -impurity

    def classification_error_impurity(self, y):
        ''' Calculates the impurity using the classification_error method
            Parameters:
            ----------
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.

            Returns:
            float <0;1>
        '''
        counts = self.class_counts(y)
        probabilities_of_label = [counts[label] / float(len(y)) for label in counts]
        return 1 - max(probabilities_of_label)

    def info_gain(self, left_node_labels, right_node_labels, current_uncertainty):
        ''' Calculates the information gain with a data split
            Parameters:
            ----------
            left_node_labels: array with samples data labels in the left child branch

            right_node_labels: array with samples data labels in the right child branch

            current_uncertainty: uncertainty of a current parent node

            Returns:
            float <0;1>
        '''
        proba = float(len(left_node_labels)) / (len(left_node_labels) + len(right_node_labels))
        return current_uncertainty - proba * self.impurity(left_node_labels) - (1 - proba) * self.impurity(right_node_labels)

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

        for feature in range(n_features):
            values = self.unique_values_with_biggest_margin(X, feature)
            for val in values:
                question = self.Question(feature, val, self)
                true_x, true_y, false_x, false_y = self.partition(X, y, question)
                if len(true_y) == 0 or len(false_y) == 0:
                    continue
                gain = self.info_gain(true_y, false_y, curr_uncertainty)
                if(gain >= best_gain):
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def build_tree(self, X, y, depth):
        ''' Builds decision tree
            Parameters:
            ----------
            X: array-like of shape (n_samples, n_features)
                    Training vectors, where `n_samples` is the number of samples and
                    `n_features` is the number of predictors.

            y : array-like of shape (n_samples,) or (n_samples, n_targets)
                    Target vectors, where `n_samples` is the number of samples and
                    `n_targets` is the number of response variables.
            depth: node depth in the tree
        '''
        if self.max_depth is not None and depth >= self.max_depth:
            return self.Leaf(self.class_counts(y))
        gain, question = self.find_best_split(X, y)
        if gain == 0:
            return self.Leaf(self.class_counts(y))
        true_x, true_y, false_x, false_y = self.partition(X, y, question)
        true_branch = self.build_tree(true_x, true_y, depth+1)
        false_branch = self.build_tree(false_x, false_y, depth+1)
        return self.DecisionNode(question, true_branch, false_branch)

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
        self.tree = self.build_tree(X, y, 0)

    def print_node(self, node, spacing=''):
        ''' Simple method to visualize tree starting from a given node
            Parameters:
            ----------
            spacing: stretches branch display
        '''
        if isinstance(node, self.Leaf):
            print(f'{spacing}Predict: {node.predictions}')
            return
        print(f'{spacing}{node.question}')
        print(f'{spacing}--> True:')
        self.print_node(node.true_branch, spacing + '\t')

        print(f'{spacing}--> False:')
        self.print_node(node.false_branch, spacing + '\t')

    def print_tree(self, spacing=''):
        ''' Simple method to visualize decision tree
            Parameters:
            ----------
            spacing: stretches branch display
        '''
        self.print_node(self.tree, spacing)

    def classify(self, X, node):
        ''' Returns number of classifications from a given node
            Parameters:
            ----------
            X: sample features

            node: tree node from which the leaf search will begin

            Returns:
            array of labels and their count in the tree leaf
        '''
        if isinstance(node, self.Leaf):
            return node.predictions
        if node.question.match(X):
            return self.classify(X, node.true_branch)
        else:
            return self.classify(X, node.false_branch)

    def single_predict(self, x):
        ''' Predicts the class of the input data (binary classification) for a single sample
            
            Parameters:
            ----------
            X: array-like of shape (n_features)
                Training vectors : `n_features` is the number of predictors.

            Return:
            int (1- or 1): Class label
        '''
        predictions = self.classify(x, self.tree)
        return max(predictions.items(), key=operator.itemgetter(1))[0]

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
        if X.ndim == 1:
            return self.single_predict(X)
        return np.array([self.single_predict(x) for x in X])

    def single_predict_proba(self, x):
        predictions = self.classify(x, self.tree)
        negative_count = predictions[-1] if -1 in predictions else 0
        positive_count = predictions[1] if 1 in predictions else 0
        count_sum = negative_count + positive_count
        return [float(negative_count / count_sum), float(positive_count / count_sum)]

    def predict_proba(self, X):
        probas = []
        for x in X:
            proba = self.single_predict_proba(x)
            probas.append(proba)
        return np.array(probas)

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
        for predicted_y, yi in zip(predicted_labels, y):
            if predicted_y == yi:
                score += 1
        return score / len(y)
        
    def get_params(self):
        return {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
        }

    def __repr__(self):
        return f'DecisionTreeClassifier(criterion={self.criterion}, max_depth={self.max_depth})'

    def __str__(self):
        return 'Decision Tree'