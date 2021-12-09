from evaluators.accuracy_evaluator import AccuracyEvaluator
from evaluators.usage_evaluator import UsageEvaluator
from evaluators.precision_evaluator import PrecisionEvaluator
from evaluators.recall_evaluator import RecallEvaluator
from evaluators.f1_score_evaluator import F1ScoreEvaluator
from evaluators.roc_evaluator import RocEvaluator
from evaluators.log_loss_evaluator import LogLossEvaluator
from evaluators.usage_evaluator import UsageEvaluator
from evaluators.learning_curve_evaluator import LearningCurveEvaluator
import utils.plots as plots
from utils.report_handler import ReportHandler
import evaluators.metrics as eval
from utils.decorators import save_plot
import numpy as np
import pandas as pd
import logging
from utils.report_handler import ReportHandler
from errors.plot_exceptions import TooManyDimensionsToPlotException
from _plotly_utils.exceptions import PlotlyError
from scipy.sparse import vstack


class Evaluator(object):

    def __init__(self, *factories):
        self.factories = factories
        self.accuracy_evaluator = AccuracyEvaluator(*factories)
        self.precision_evaluator = PrecisionEvaluator(*factories)
        self.recall_evaluator = RecallEvaluator(*factories)
        self.f1_score_evaluator = F1ScoreEvaluator(*factories)
        self.roc_evaluator = RocEvaluator(*[factory for factory in self.factories if hasattr(factory.classifier, 'predict_proba')])
        self.log_loss_evaluator = LogLossEvaluator(*[factory for factory in self.factories if hasattr(factory.classifier, 'predict_proba')])
        self.usage_evaluator = None
        self.learning_curve_evaluator = LearningCurveEvaluator(*factories)

    @save_plot
    def plot_learning_accuracy(self, short_labels=False):
        return self.accuracy_evaluator.plot_learning_accuracy(short_labels=short_labels)

    @save_plot
    def plot_test_accuracy(self, short_labels=False):
        return self.accuracy_evaluator.plot_test_accuracy(short_labels=short_labels)

    @save_plot
    def plot_precision_scores(self, short_labels=False):
        return self.precision_evaluator.plot_precision_scores(short_labels=short_labels)

    @save_plot
    def plot_recall_scores(self, short_labels=False):
        return self.recall_evaluator.plot_recall_scores(short_labels=short_labels)

    @save_plot
    def plot_f1_scores(self, short_labels=False):
        return self.f1_score_evaluator.plot_f1_scores(short_labels=short_labels)

    @save_plot
    def plot_confusion_matrix(self, factory):
        confmat = eval.generate_confusion_matrix(factory.classifier.predict(factory.X_test), factory.y_test)
        return plots.plot_confusion_matrix(confmat, f'{repr(factory.classifier)} Confusion Matrix')

    def plot_confusion_matrixes(self, save_plots=False):
        for factory in self.factories:
            if save_plots:
                file_name = f'{str(factory.classifier)}_confmat'
                file_name = file_name.replace(' ', '_').lower()
                self.plot_confusion_matrix(factory, file_name=file_name)
            else:
                self.plot_confusion_matrix(factory)

    def get_confusion_matrixes(self):
        conf_mats = []
        for factory in self.factories:
            try:
                confmat = eval.generate_confusion_matrix(factory.classifier.predict(factory.X_test), factory.y_test)
                conf_mats.append(plots.plot_confusion_matrix(confmat, f'{repr(factory.classifier)} Confusion Matrix'))
            except PlotlyError:
                return []
        return conf_mats

    @save_plot
    def plot_roc_curve(self, short_labels=False):
        return self.roc_evaluator.plot_roc_curve(short_labels=short_labels)

    @save_plot
    def plot_log_loss(self, short_labels=False):
        return self.log_loss_evaluator.plot_log_loss(short_labels=short_labels)

    @save_plot
    def plot_decision_region(self, X, y, classifier, title, resolution=0.02, data_dim=2):
        return plots.get_decision_regions_plot(X, y, classifier, title, resolution=resolution)

    def plot_decision_regions(self, data='all', resolution=0.02, save_plots=False):
        data_samples = ['train', 'test', 'all', '']
        if not data in data_samples:
            raise ValueError(f'Wrong value: data must be in {data_samples}')

        for factory in self.factories:
            title = f"{str(factory.classifier)}'s Decision Regions"
            if data == 'train':
                X = factory.X_train
                y = factory.y_train
            elif data == 'test':
                X = factory.X_test
                y = factory.y_test
            else:
                try:
                    X = np.concatenate((factory.X_train, factory.X_test))
                except ValueError:
                    # ValueError: zero-dimensional arrays cannot be concatenated: sparse matrices
                    X = vstack((factory.X_train, factory.X_test))
                y = np.concatenate((factory.y_train, factory.y_test))

            if save_plots:
                file_name = f'{repr(factory.classifier)}_decision_regions'
                file_name = file_name.replace(' ', '_').lower()
                self.plot_decision_region(X, y, factory.classifier, title, resolution=resolution, file_name=file_name, data_dim=X.shape[1])
            else:
                plots.plot_decision_regions(X, y, factory.classifier, title, resolution=resolution)

    def get_decision_regions(self):
        figs = []
        for factory in self.factories:
            try:
                X = np.concatenate((factory.X_train, factory.X_test))
                y = np.concatenate((factory.y_train, factory.y_test))
                figs.append(plots.get_decision_regions_plot(X, y, factory.classifier, f"{str(factory.classifier)}'s Decision Regions", resolution=0.05))
            except ValueError:
                # ValueError: zero-dimensional arrays cannot be concatenated: sparse matrices
                X = vstack((factory.X_train, factory.X_test))
            except TooManyDimensionsToPlotException as exception:
                logging.warning(exception)
        return figs

    def plot_learning_curves(self, short_labels=False):
        return self.learning_curve_evaluator.plot_learning_curve(short_labels=short_labels)

    def add_usages(self, usages):
        self.usage_evaluator = UsageEvaluator(usages)

    def init_report(self):
        report_elements = ReportHandler.ReportElements(
            data_correlation=plots.get_feature_correlation(
                np.concatenate((self.factories[0].X_train, self.factories[0].X_test)),
                np.concatenate((self.factories[0].y_train, self.factories[0].y_test))
            ) if self.factories[0].X_train.shape[1] < 20 else None,
            usage=[
                self.usage_evaluator.plot_time_usage(),
                self.usage_evaluator.plot_memory_usage(),
            ],
            learning_curves=self.learning_curve_evaluator.plot_learning_curve(),
            accuracy=[
                self.accuracy_evaluator.plot_learning_accuracy(),
                self.accuracy_evaluator.plot_test_accuracy(),
            ],
            precision=self.precision_evaluator.plot_precision_scores(),
            recall=self.recall_evaluator.plot_recall_scores(),
            f1=self.f1_score_evaluator.plot_f1_scores(),
            roc=self.roc_evaluator.plot_roc_curve(),
            log_loss=self.log_loss_evaluator.plot_log_loss(),
            confusion_matrixes=self.get_confusion_matrixes(),
            decision_regions=self.get_decision_regions()
        )
        
        params = {}
        for factory in self.factories:
            data = [[key, val] for key, val in factory.classifier.get_params().items()]
            params[str(factory.classifier)] = pd.DataFrame(data=data, columns=['Parameter', 'Value'])
        
        self.report_handler = ReportHandler(report_elements, params)

    def get_report_elements(self, app):
        return self.report_handler.get_report_elements(app)