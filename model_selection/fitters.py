from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import evaluators.metrics as eval
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
from errors.factory_exceptions import FitterWithoutClassifierException
from sklearn.preprocessing import StandardScaler

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

class ClassifierFitter(ABC):

    def prepare_data(self, X_train, X_test, y_train, y_test):
        ''' Standarizes data and splits it into training and test datasets'''
        return X_train, X_test, y_train, y_test

    def prepare_standarized_data(self, X_train, X_test, y_train, y_test):
        try:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        except ValueError:
            # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead.
            sc = StandardScaler(with_mean=False)
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        return X_train, X_test, y_train, y_test

    def prepare_params(self, Classifier, X, y, **kwargs):
        best_params = eval.grid_search(Classifier, kwargs, X, y)
        return best_params[0]

    @abstractmethod
    def get_best_params(self, X, y, **kwargs):
        ''' Finds best parameters using grid search'''


class PerceptronFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(Perceptron, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class AdalineFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(Adaline, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class LogisticRegressionFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(LogisticRegression, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SupportVectorClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SupportVectorClassifier, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class GausianNaiveBayesClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(GausianNaiveBayesClassifier, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class KNearestNeighboursFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(KNearestNeighbours, X, y, **kwargs)


class DecisionTreeClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(DecisionTreeClassifier, X, y, **kwargs)


class RandomForrestClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(RandomForrestClassifier, X, y, **kwargs)


class NeuralNetMLPFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(NeuralNetMLP, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class AdaBoostFitter(ClassifierFitter):

    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(AdaBoost, X, y, **kwargs)


class SklearnPerceptronFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnPerceptron, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SklearnDecisionTreeClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnDecisionTreeClassifier, X, y, **kwargs)


class SklearnSGDClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnSGDClassifier, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SklearnKNeighborsClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnKNeighborsClassifier, X, y, **kwargs)


class SklearnLogisticRegressionFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnLogisticRegression, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SklearnGaussianNBFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnGaussianNB, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SklearnRandomForestClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnRandomForestClassifier, X, y, **kwargs)


class SklearnMLPClassifierFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnMLPClassifier, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SklearnSVCFitter(ClassifierFitter):
    
    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnSVC, X, y, **kwargs)

    def prepare_data(self, X_train, X_test, y_train, y_test):
        return super().prepare_standarized_data(X_train, X_test, y_train, y_test)


class SklearnAdaBoostClassifierFitter(ClassifierFitter):

    def get_best_params(self, X, y, **kwargs):
        return super().prepare_params(SklearnAdaBoostClassifier, X, y, **kwargs)


def get_fitter(Classifier):
    
    fitters = {
        Perceptron: PerceptronFitter(),
        Adaline: AdalineFitter(),
        LogisticRegression: LogisticRegressionFitter(),
        SupportVectorClassifier: SupportVectorClassifierFitter(),
        GausianNaiveBayesClassifier: GausianNaiveBayesClassifierFitter(),
        KNearestNeighbours: KNearestNeighboursFitter(),
        DecisionTreeClassifier: DecisionTreeClassifierFitter(),
        RandomForrestClassifier: RandomForrestClassifierFitter(),
        NeuralNetMLP: NeuralNetMLPFitter(),
        AdaBoost: AdaBoostFitter(),
        SklearnPerceptron: SklearnPerceptronFitter(),
        SklearnDecisionTreeClassifier: SklearnDecisionTreeClassifierFitter(),
        SklearnSGDClassifier: SklearnSGDClassifierFitter(),
        SklearnKNeighborsClassifier: SklearnKNeighborsClassifierFitter(),
        SklearnLogisticRegression: SklearnLogisticRegressionFitter(),
        SklearnGaussianNB: SklearnGaussianNBFitter(),
        SklearnRandomForestClassifier: SklearnRandomForestClassifierFitter(),
        SklearnMLPClassifier: SklearnMLPClassifierFitter(),
        SklearnSVC: SklearnSVCFitter(),
        SklearnAdaBoostClassifier: SklearnAdaBoostClassifierFitter()
    }

    if Classifier in fitters:
        return fitters[Classifier]
    else:
        raise FitterWithoutClassifierException(Classifier)