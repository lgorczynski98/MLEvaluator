from model_selection.factory import get_factory
from utils.plots import get_plot_histogram
import sklearn.metrics as metrics


class LogLossEvaluator(object):

    def __init__(self, *factories):
        self.factories = factories

    def plot_log_loss(self, short_labels=False):
        values = []
        xlabels = []
        for factory in self.factories:
            
            if not hasattr(factory.classifier, 'predict_proba'):
                continue
            
            y_pred_proba = factory.classifier.predict_proba(factory.X_test)
            log_loss = metrics.log_loss(factory.y_test, y_pred_proba)
            values.append(log_loss)
            xlabels.append(str(factory.classifier) if short_labels else repr(factory.classifier))
        
        return get_plot_histogram(values=values,
                        xlabels=xlabels,
                        ylabel='Log Loss',
                        title='Log Loss Comparison',
        )