from model_selection.model_selector import ModelSelector
import data_generators.sample_generators as generators
from sklearn.datasets import load_digits
import logging
import logging.config
import numpy as np
import pandas as pd
import time
import os
from app.Report import Report

from models.perceptron import Perceptron
from models.adaline_gradient_desc import Adaline
from models.NeuralNetMLP import NeuralNetMLP
from models.k_nearest_neighbours import KNearestNeighbours
from models.svm import SupportVectorClassifier
from models.decision_tree import DecisionTreeClassifier
from models.random_forrest import RandomForrestClassifier
from models.logistic_regression import LogisticRegression
from models.naive_bayes import GausianNaiveBayesClassifier
from models.adaboost import AdaBoost

from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.svm import SVC as SklearnSVC
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier


def main():

    logging.config.fileConfig('log/logging.conf')
    
    logging.info(f'Starting {__name__}')

    report = Report()

    ######################################BLOBS 2D DATASET###################################################

    X, y = generators.make_blobs(n_samples=200, centers=2, n_features=2, random_state=56, cluster_std=2)
    y = np.where(y == 1, 1, -1)
 
    df = pd.DataFrame(data=X, columns=[f'X{idx}' for idx in range(X.shape[1])])
    df['Y'] = y

    blobs2d_model_selector = ModelSelector(
                # {'classifier': Perceptron, 'eta': [0.01, 0.001], 'n_iter': [10, 15]},
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                # {'classifier': Adaline, 'eta': [0.01, 0.001], 'n_iter': [30, 50]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                # {'classifier': SupportVectorClassifier, 'eta': [0.1, 0.001], 'n_iter': [100], 'c': [1000, 10000], 'kernel': [None, 'polynomial'], 'd': [2], 'r': [1]},
                {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                # {'classifier': KNearestNeighbours, 'k': [3, 5], 'metric': ['euclidean', 'manhattan']},
                {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
                # {'classifier': GausianNaiveBayesClassifier},
                {'classifier': SklearnGaussianNB},
                # {'classifier': LogisticRegression, 'eta': [0.01, 0.001], 'n_iter': [10, 100], 'regularization': ['', 'l2']},
                {'classifier': SklearnLogisticRegression, 'C': [0.01, 0.001], 'max_iter': [10, 100], 'penalty': ['none', 'l2']},
                # {'classifier': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [None, 4]},
                {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini', 'entropy'], 'max_depth':  [None, 3]},
                # {'classifier': RandomForrestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                # {'classifier': AdaBoost, 'classifier_cls': [DecisionTreeClassifier], 'params': [{'criterion': 'gini', 'max_depth': 1}], 'n_estimators': [50]},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    blobs2d_model_selector.fit(X, y, save_factories=False)

    blobs2d_model_selector.evaluator.init_report()
    blobs2d_model_selector.evaluator.report_handler.sample_data = df
    report.add_page('Blobs_2D', blobs2d_model_selector.evaluator.get_report_elements(report.app))
    blobs2d_model_selector.save('blob2d.pkl')


######################################BLOBS 3D DATASET###################################################

    X, y = generators.make_blobs(n_samples=500, centers=2, n_features=3, random_state=2, cluster_std=2)
    y = np.where(y == 1, 1, -1)
 
    df = pd.DataFrame(data=X, columns=[f'X{idx}' for idx in range(X.shape[1])])
    df['Y'] = y

    blobs3d_model_selector = ModelSelector(
                # {'classifier': Perceptron, 'eta': [0.01, 0.001], 'n_iter': [10, 15]},
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                # {'classifier': Adaline, 'eta': [0.01, 0.001], 'n_iter': [30, 50]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                # {'classifier': SupportVectorClassifier, 'eta': [0.1, 0.001], 'n_iter': [100], 'c': [1000, 10000], 'kernel': [None, 'polynomial'], 'd': [2], 'r': [1]},
                {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                # {'classifier': KNearestNeighbours, 'k': [3, 5], 'metric': ['euclidean', 'manhattan']},
                {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
                # {'classifier': GausianNaiveBayesClassifier},
                {'classifier': SklearnGaussianNB},
                # {'classifier': LogisticRegression, 'eta': [0.01, 0.001], 'n_iter': [10, 100], 'regularization': ['', 'l2']},
                {'classifier': SklearnLogisticRegression, 'C': [0.01, 0.001], 'max_iter': [10, 100], 'penalty': ['none', 'l2']},
                # {'classifier': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [None, 4]},
                {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini', 'entropy'], 'max_depth':  [None, 3]},
                # {'classifier': RandomForrestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                # {'classifier': AdaBoost, 'classifier_cls': [DecisionTreeClassifier], 'params': [{'criterion': 'gini', 'max_depth': 1}], 'n_estimators': [50]},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    blobs3d_model_selector.fit(X, y, save_factories=False)

    blobs3d_model_selector.evaluator.init_report()
    blobs3d_model_selector.evaluator.report_handler.sample_data = df
    report.add_page('Blobs_3D', blobs3d_model_selector.evaluator.get_report_elements(report.app))
    blobs3d_model_selector.save('blob3d.pkl')


######################################BLOBS MANY FEATURES DATASET###################################################

    X, y = generators.make_blobs(n_samples=2000, centers=2, n_features=8, random_state=1, cluster_std=2)
    y = np.where(y == 1, 1, -1)
 
    df = pd.DataFrame(data=X, columns=[f'X{idx}' for idx in range(X.shape[1])])
    df['Y'] = y

    many_features_blobs_model_selector = ModelSelector(
                # {'classifier': Perceptron, 'eta': [0.01, 0.001], 'n_iter': [10, 15]},
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                # {'classifier': Adaline, 'eta': [0.01, 0.001], 'n_iter': [30, 50]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                # {'classifier': SupportVectorClassifier, 'eta': [0.1, 0.001], 'n_iter': [100], 'c': [1000, 10000], 'kernel': [None, 'polynomial'], 'd': [2], 'r': [1]},
                {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                # {'classifier': KNearestNeighbours, 'k': [3, 5], 'metric': ['euclidean', 'manhattan']},
                {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
                # {'classifier': GausianNaiveBayesClassifier},
                {'classifier': SklearnGaussianNB},
                # {'classifier': LogisticRegression, 'eta': [0.01, 0.001], 'n_iter': [10, 100], 'regularization': ['', 'l2']},
                {'classifier': SklearnLogisticRegression, 'C': [0.01, 0.001], 'max_iter': [10, 100], 'penalty': ['none', 'l2']},
                # {'classifier': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [None, 4]},
                {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini', 'entropy'], 'max_depth':  [None, 3]},
                # {'classifier': RandomForrestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                # {'classifier': AdaBoost, 'classifier_cls': [DecisionTreeClassifier], 'params': [{'criterion': 'gini', 'max_depth': 1}], 'n_estimators': [50]},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    many_features_blobs_model_selector.fit(X, y, save_factories=False)

    many_features_blobs_model_selector.evaluator.init_report()
    many_features_blobs_model_selector.evaluator.report_handler.sample_data = df
    report.add_page('Many_Features_Blobs', many_features_blobs_model_selector.evaluator.get_report_elements(report.app))
    many_features_blobs_model_selector.save('blob_many_features.pkl')


    ######################################MOONS DATASET###################################################

    X, y = generators.generate_moons(n_samples=500, noise=0.1)
    y = np.where(y == 1, 1, -1)
 
    df = pd.DataFrame(data=X, columns=[f'X{idx}' for idx in range(X.shape[1])])
    df['Y'] = y

    moons_model_selector = ModelSelector(
                # {'classifier': Perceptron, 'eta': [0.01, 0.001], 'n_iter': [10, 15]},
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                # {'classifier': Adaline, 'eta': [0.01, 0.001], 'n_iter': [30, 50]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                # {'classifier': SupportVectorClassifier, 'eta': [0.1, 0.001], 'n_iter': [100], 'c': [1000, 10000], 'kernel': [None, 'polynomial'], 'd': [2], 'r': [1]},
                {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                # {'classifier': KNearestNeighbours, 'k': [3, 5], 'metric': ['euclidean', 'manhattan']},
                {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
                # {'classifier': GausianNaiveBayesClassifier},
                {'classifier': SklearnGaussianNB},
                # {'classifier': LogisticRegression, 'eta': [0.01, 0.001], 'n_iter': [10, 100], 'regularization': ['', 'l2']},
                {'classifier': SklearnLogisticRegression, 'C': [0.01, 0.001], 'max_iter': [10, 100], 'penalty': ['none', 'l2']},
                # {'classifier': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [None, 4]},
                {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini', 'entropy'], 'max_depth':  [None, 3]},
                # {'classifier': RandomForrestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                # {'classifier': AdaBoost, 'classifier_cls': [DecisionTreeClassifier], 'params': [{'criterion': 'gini', 'max_depth': 1}], 'n_estimators': [50]},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    moons_model_selector.fit(X, y, save_factories=False)

    moons_model_selector.evaluator.init_report()
    moons_model_selector.evaluator.report_handler.sample_data = df
    report.add_page('Moons', moons_model_selector.evaluator.get_report_elements(report.app))
    moons_model_selector.save('moons.pkl')
    
#     ######################################XOR DATASET###################################################
    
    X, y = generators.generate_batch_xor(2*300, sigma=0.25)
    y = np.where(y == 1, 1, -1)
 
    df = pd.DataFrame(data=X, columns=[f'X{idx}' for idx in range(X.shape[1])])
    df['Y'] = y

    xor_model_selector = ModelSelector(
                # {'classifier': Perceptron, 'eta': [0.01, 0.001], 'n_iter': [10, 15]},
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                # {'classifier': Adaline, 'eta': [0.01, 0.001], 'n_iter': [30, 50]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                # {'classifier': SupportVectorClassifier, 'eta': [0.1, 0.001], 'n_iter': [100], 'c': [1000, 10000], 'kernel': [None, 'polynomial'], 'd': [2], 'r': [1]},
                {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                # {'classifier': KNearestNeighbours, 'k': [3, 5], 'metric': ['euclidean', 'manhattan']},
                {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
                # {'classifier': GausianNaiveBayesClassifier},
                {'classifier': SklearnGaussianNB},
                # {'classifier': LogisticRegression, 'eta': [0.01, 0.001], 'n_iter': [10, 100], 'regularization': ['', 'l2']},
                {'classifier': SklearnLogisticRegression, 'C': [0.01, 0.001], 'max_iter': [10, 100], 'penalty': ['none', 'l2']},
                # {'classifier': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [None, 4]},
                {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini', 'entropy'], 'max_depth':  [None, 3]},
                # {'classifier': RandomForrestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                # {'classifier': AdaBoost, 'classifier_cls': [DecisionTreeClassifier], 'params': [{'criterion': 'gini', 'max_depth': 1}], 'n_estimators': [50]},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    xor_model_selector.fit(X, y, save_factories=False)

    xor_model_selector.evaluator.init_report()
    xor_model_selector.evaluator.report_handler.sample_data = df
    report.add_page('XOR', xor_model_selector.evaluator.get_report_elements(report.app))
    xor_model_selector.save('xor.pkl')
    

######################################DIGITS DATASET###################################################
    
    digits = load_digits()
    X, y = digits.data, digits.target

    df = pd.DataFrame(data=X, columns=[f'X{idx}' for idx in range(X.shape[1])])
    df['Y'] = y

    digits_model_selector = ModelSelector(
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3, 5], 'metric': ['euclidean', 'manhattan']},
                {'classifier': SklearnGaussianNB},
                {'classifier': SklearnLogisticRegression, 'C': [0.01, 0.001], 'max_iter': [10, 100], 'penalty': ['none', 'l2']},
                {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini', 'entropy'], 'max_depth':  [None, 3]},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    digits_model_selector.fit(X, y, save_factories=False)

    digits_model_selector.evaluator.init_report()
    digits_model_selector.evaluator.report_handler.sample_data = df
    report.add_page('Digits', digits_model_selector.evaluator.get_report_elements(report.app))
    digits_model_selector.save('digits.pkl')
    

######################################IMDB SENTIMENT ANALYZIS DATASET###################################################

    X, y = generators.generate_imdb_dataset()

    imdb_model_selector = ModelSelector(
                # {'classifier': Perceptron, 'eta': [0.01, 0.001], 'n_iter': [10, 15]},
                {'classifier': SklearnPerceptron, 'eta0':  [0.01, 0.001], 'max_iter':  [10, 15]},
                # {'classifier': Adaline, 'eta': [0.01, 0.001], 'n_iter': [30, 50]},
                {'classifier': SklearnSGDClassifier, 'eta0': [0.01, 0.001], 'max_iter': [30, 50]},
                # {'classifier': SupportVectorClassifier, 'eta': [0.1, 0.001], 'n_iter': [100], 'c': [1000, 10000], 'kernel': [None, 'polynomial'], 'd': [2], 'r': [1]},
                # {'classifier': SklearnSVC, 'max_iter': [100], 'C': [1000, 10000], 'kernel': ['linear', 'poly'], 'degree': [2]},
                # {'classifier': KNearestNeighbours, 'k': [3, 5], 'metric': ['euclidean', 'manhattan']},
                # {'classifier': SklearnKNeighborsClassifier, 'n_neighbors': [3], 'metric': ['euclidean']},
                # {'classifier': GausianNaiveBayesClassifier},
                # {'classifier': SklearnGaussianNB},
                # {'classifier': LogisticRegression, 'eta': [0.01, 0.001], 'n_iter': [10, 100], 'regularization': ['', 'l2']},
                # {'classifier': SklearnLogisticRegression, 'C': [0.001], 'max_iter': [100], 'penalty': ['none']},
                # {'classifier': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [None, 4]},
                # {'classifier': SklearnDecisionTreeClassifier, 'criterion':  ['gini'], 'max_depth':  [5]},
                # {'classifier': RandomForrestClassifier, 'n_estimators': [3, 5], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 4], 'max_features': ['sqrt']},
                {'classifier': SklearnRandomForestClassifier, 'n_estimators': [5], 'criterion': ['gini'], 'max_depth': [5], 'max_features': ['sqrt']},
                # {'classifier': AdaBoost, 'classifier_cls': [DecisionTreeClassifier], 'params': [{'criterion': 'gini', 'max_depth': 1}], 'n_estimators': [50]},
                # {'classifier': SklearnAdaBoostClassifier, 'base_estimator': [SklearnDecisionTreeClassifier(criterion='gini', max_depth=1)], 'n_estimators': [50]},
        )
    imdb_model_selector.fit(X, y, save_factories=False)

    imdb_model_selector.evaluator.init_report()
    report.add_page('IMDB', imdb_model_selector.evaluator.get_report_elements(report.app))
    imdb_model_selector.save('imdb.pkl')


    report.prepare_elements()
    report.run()

if __name__ == '__main__':
    main()