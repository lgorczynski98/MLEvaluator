from utils.plots import get_roc_curves_plot
import sklearn.metrics as metrics

class RocEvaluator(object):

    def __init__(self, *factories):
        self.factories = factories

    def plot_roc_curve(self, short_labels=False):
        classifiers, fprs, tprs, aucs = [], [], [], []
        for factory in self.factories:
            if not hasattr(factory.classifier, 'predict_proba'):
                continue
            classifiers.append(str(factory.classifier) if short_labels else repr(factory.classifier))
            y_pred_proba = factory.classifier.predict_proba(factory.X_test)[::, 1]
            try:
                fpr, tpr, _ = metrics.roc_curve(factory.y_test, y_pred_proba)
            except ValueError:
                # ValueError: multiclass format is not supported
                return None
            fprs.append(fpr)
            tprs.append(tpr)
            auc = metrics.roc_auc_score(factory.y_test, y_pred_proba)
            aucs.append(auc)

        return get_roc_curves_plot(classifiers, fprs, tprs, aucs)