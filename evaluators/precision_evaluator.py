from numpy.lib.function_base import average
from model_selection.factory import get_factory
from utils.plots import get_plot_histogram
import evaluators.metrics as eval
import sklearn.metrics as metrics
import numpy as np

class PrecisionEvaluator(object):

    def __init__(self, *factories):
        self.factories = factories

    def plot_precision_scores(self, short_labels=False):
        is_binary = len(np.unique(self.factories[0].y_test)) == 2
        return get_plot_histogram(values=[eval.get_precision_score(factory.classifier.predict(factory.X_test), factory.y_test) if is_binary
                            else metrics.precision_score(factory.classifier.predict(factory.X_test), factory.y_test, average='micro') for factory in self.factories],
                        xlabels=[str(factory.classifier) if short_labels else repr(factory.classifier) for factory in self.factories],
                        ylabel='Precision',
                        title='Precision Comparison',
        )