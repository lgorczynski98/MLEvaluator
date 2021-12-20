import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import itertools
import logging
import multiprocessing
from multiprocessing import Process

from models.classifier import Classifier


def dict_cartesian_product(params):
	keys = params.keys()
	vals = params.values()
	combinations = []
	for instance in itertools.product(*vals):
		combinations.append(dict(zip(keys, instance)))
	return combinations

def grid_search(classifier_cls, params, X, y, n_splits=10, random_state=None, shuffle=False):
	combinations = dict_cartesian_product(params)
	checked_combinations = []
	scores = []
	logging.warning(f'{classifier_cls} grid search')
	for combination in combinations:
		classifier = classifier_cls(**combination)
		
		if isinstance(classifier, Classifier):
			if classifier.get_effective_params() in checked_combinations:
				scores.append(0)
				logging.debug(f'Skipping combination {combination} because the same effective combination was already checked')
				continue
			else:
				checked_combinations.append(classifier.get_effective_params())

		mean, std = stratified_k_fold(classifier, X, y)
		scores.append(mean)
		logging.info(f'{combination}: mean: {mean} std: {std}')

	m = max(scores)
	idx = [i for i, j in enumerate(scores) if j == m]

	logging.warning('Best score:')
	for i in idx:
		logging.warning(f'{combinations[i]}: {scores[i]}')
	return [combinations[i] for i in idx]

def single_multiprocessing_fit(classifier_cls, params, x_train, x_test, y_train, y_test, scores, subset):
	classifier = classifier_cls(**params)
	classifier.fit(x_train, y_train)
	score = classifier.score(x_test, y_test)
	scores.append(score)
	return {'classifier': classifier, 'subset': subset, 'score': score}

def proc_callback(result):
	classifier = result['classifier']
	subset = result['subset']
	score = result['score']
	logging.debug(f'{repr(classifier)}: subset {subset} - score: {score:.4f}')

def perform_multiprocessed_stratified_k_fold(classifier, x_train, y_train, n_splits=10, random_state=None, shuffle=False):
	kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle).split(x_train, y_train)
	manager = multiprocessing.Manager()
	scores = manager.list()
	classifier_cls = type(classifier)
	params = classifier.get_params()
	pool = multiprocessing.Pool(n_splits)
	results = []
	for k, (train, test) in enumerate(kfold):
		r = pool.apply_async(single_multiprocessing_fit, (classifier_cls, params, x_train[train], x_train[test], y_train[train], y_train[test], scores, k+1))
		results.append(r)

	for r in results:
		r.wait()
	pool.close()
	pool.join()
	for idx, score in enumerate(scores):
		logging.debug(f'{repr(classifier)}: subset {idx+1} - score: {score:.4f}')

	mean, std = np.mean(scores), np.std(scores)
	logging.info(f'{repr(classifier)}: mean {mean:.4f} +/- {std:.4f}')
	return mean, std

def perform_stratified_k_fold(classifier, x_train, y_train, n_splits=10, random_state=None, shuffle=False):
	kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle).split(x_train, y_train)
	scores = []
	for k, (train, test) in enumerate(kfold):
		classifier.fit(x_train[train], y_train[train])
		score = classifier.score(x_train[test], y_train[test])
		scores.append(score)
		logging.debug(f'{repr(classifier)}: subset {k+1} - score: {score:.4f}')
		
	mean, std = np.mean(scores), np.std(scores)
	logging.info(f'{repr(classifier)}: mean {mean:.4f} +/- {std:.4f}')
	return mean, std

def stratified_k_fold(classifier, x_train, y_train, n_splits=10, random_state=None, shuffle=False, multiprocessed=False):
	''' Stratified K Fold calculation with sklearn StratifiedKFold usage.

        Parameters:
        ----------
        classifier: Classifier implementing fit function
        
        x_train: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

        y_train: Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target vectors, where `n_samples` is the number of samples and
                `n_targets` is the number of response variables.

        n_splits: Number of performed data splits

        random_state: sklearn StratifiedKFold random_state param

        shuffle: sklearn StratifiedKFold shuffle param

        Returns:
        float: mean - mean value of the stratified K fold algorithm
        float: std - standard deviation of the stratified K fold algorithm
     '''
	if multiprocessed:
		return perform_multiprocessed_stratified_k_fold(classifier, x_train, y_train, n_splits, random_state, shuffle)
	else:
		return perform_stratified_k_fold(classifier, x_train, y_train, n_splits, random_state, shuffle)

def cross_validation_score(classifier, x_train, y_train, cv=10, n_jobs=1):
	'''Sklear cross validation. Accepts only scikit-learn estimators

        Parameters:
        ----------
        classifier: Classifier implementing fit function
        
        x_train: array-like of shape (n_samples, n_features)
                Training vectors, where `n_samples` is the number of samples and
                `n_features` is the number of predictors.

        y_train: Y : array-like of shape (n_samples,) or (n_samples, n_targets)
                Target vectors, where `n_samples` is the number of samples and
                `n_targets` is the number of response variables.

        cv: Number of performed data splits

        n_jobs: Number of processors taking part in calculations
                int(-1) value: All available processors
    '''
	scores = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=cv, n_jobs=n_jobs)
	logging.warning(f'Scores: {scores}')
	logging.warning(f'Score accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

def generate_confusion_matrix(y_predicted, y):
	''' Generates confusion matrix.
		
		Parameters:
		----------
		y_predicted: Vector of predicted labels

		y: Vector of target labels

		Returns:
		Confusion matrix - matrix structure:
                                [[True False, False Positive][False Negative, True Positive]]
		
	'''
	confmat = confusion_matrix(y_true=y, y_pred=y_predicted)
	return confmat

def get_precision_score(y_predicted, y):
	''' Calculates precision score metric.
		
		Parameters:
		----------
		y_predicted: Vector of predicted labels

		y: Vector of target labels

		Returns:
		float: Precision score
	'''
	return precision_score(y_true=y, y_pred=y_predicted)

def get_recall_score(y_predicted, y):
	''' Calculates recall score metric.
		
		Parameters:
		----------
		y_predicted: Vector of predicted labels

		y: Vector of target labels

		Returns:
		float: Recall score
	'''
	return recall_score(y_true=y, y_pred=y_predicted)

def get_f1_score(y_predicted, y):
	''' Calculates F1 score metric.
		
		Parameters:
		----------
		y_predicted: Vector of predicted labels

		y: Vector of target labels

		Returns:
		float: F1 score
	'''
	return f1_score(y_true=y, y_pred=y_predicted)

def get_sklearn_learning_curve(classifier, X_train, y_train, cv=10, train_sizes=(0.1, 1.0, 10), n_jobs=1):
    ''' Determines cross-validated training and test scores for different training set sizes.
        Classifier must be a sklearn classifier implementation

        Parameters:
        ----------
        classifier: Sklearn classifier

        X_train: array_like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
                Target relative to X for classification or regression; None for unsupervised learning.

        cv: Number of performed data splits

        train_sizes: array-like of shape (n_ticks,), default=np.linspace(0.1, 1.0, 5)
                Relative or absolute numbers of training examples that will be used to generate the learning curve.
                If the dtype is float, it is regarded as a fraction of the maximum size of the training set
                (that is determined by the selected validation method), i.e. it has to be within (0, 1].
                Otherwise it is interpreted as absolute sizes of the training sets. 
                Note that for classification the number of samples usually have to be big enough to contain at least one sample from each class.

        n_jobs: Number of processors taking part in calculations
                int(-1) value: All available processors

        Returns:
        train_sizes: of shape (n_unique_ticks,)
                Numbers of training examples that has been used to generate the learning curve. Note that the number of ticks might be less than n_ticks because duplicate entries will be removed.

        train_scores: of shape (n_ticks, n_cv_folds)
                Scores on training sets.

        test_scores: of shape (n_ticks, n_cv_folds)
                Scores on test set.
    '''
    train_sizes, train_score, test_score = learning_curve(estimator=classifier,
                                                        X=X_train, y=y_train,
                                                        train_sizes=np.linspace(*train_sizes),
                                                        cv=cv, n_jobs=n_jobs)
    return train_sizes, train_score, test_score

def get_learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 0.95, 5), folds=5):
    cls = type(estimator)
    params = estimator.get_params()
    train_scores = []
    test_scores = []
    for train_size in train_sizes:
        train_score = []
        test_score = []
        for _ in range(folds):
            classifier = cls(**params)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
            classifier.fit(X_train, y_train)
            train_score.append(classifier.score(X_train, y_train))
            test_score.append(classifier.score(X_test, y_test))
        train_scores.append(train_score)
        test_scores.append(test_score)
    print(train_scores)
    print(test_scores)
    return train_sizes, np.array(train_scores), np.array(test_scores)