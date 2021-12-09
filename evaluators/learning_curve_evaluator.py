import evaluators.metrics as metrics
import utils.plots as plots
import numpy as np
from scipy.sparse import vstack

class LearningCurveEvaluator(object):

    def __init__(self, *factories):
        self.factories = factories

    def plot_learning_curve(self, short_labels=False):
        learning_curves = []
        for factory in self.factories:
            try:
                X = np.concatenate((factory.X_train, factory.X_test))
            except ValueError:
                # ValueError: zero-dimensional arrays cannot be concatenated: sparse matrices
                X = vstack((factory.X_train, factory.X_test))
            y = np.concatenate((factory.y_train, factory.y_test))
            train_sizes, train_scores, test_scores = metrics.learning_curve(factory.classifier, X, y)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            learning_curve = plots.get_learning_curve_plot(
                train_sizes=train_sizes,
                train_scores_mean=train_scores_mean,
                train_scores_std=train_scores_std,
                test_scores_mean=test_scores_mean,
                test_scores_std=test_scores_std,
                title=f'{str(factory.classifier)} Learning Curve' if short_labels else f'{repr(factory.classifier)} Learning Curve'
            )
            learning_curves.append(learning_curve)
        return learning_curves
