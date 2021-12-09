from utils.plots import get_plot_histogram


class AccuracyEvaluator(object):

    def __init__(self, *factories):
        self.factories = factories

    def plot_learning_accuracy(self, short_labels=False):
        return get_plot_histogram(values=[factory.classifier.score(factory.X_train, factory.y_train) for factory in self.factories],
                        xlabels=[str(factory.classifier) if short_labels else repr(factory.classifier) for factory in self.factories],
                        ylabel='Training Accuracy',
                        title='Training Accuracy Comparison',
                        )

    def plot_test_accuracy(self, short_labels=False):
        return get_plot_histogram(values=[factory.classifier.score(factory.X_test, factory.y_test) for factory in self.factories],
                        xlabels=[str(factory.classifier) if short_labels else repr(factory.classifier) for factory in self.factories],
                        ylabel='Test Accuracy',
                        title='Test Accuracy Comparison',
                        )

