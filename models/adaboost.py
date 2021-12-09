import numpy as np
import pandas as pd
from models.classifier import Classifier
from dataclasses import dataclass

class AdaBoost(Classifier):

    @dataclass
    class Estimator():
        alpha: float
        classifier: Classifier


    def __init__(self, classifier_cls, params, n_estimators):
        self.classifier_cls = classifier_cls
        self.params = params
        self.n_estimators = n_estimators

        self.estimators = [AdaBoost.Estimator(1, classifier_cls(**params)) for _ in range(self.n_estimators)]
        self.alpha_sum = 0

    def fit(self, X, y):
        n_features = X.shape[1]
        n_samples = len(y)
        df = pd.DataFrame(data=X)
        df['y'] = y
        df['predicted_y'] = 0
        df['is_incorrect'] = False
        df['weight'] = 1 / n_samples
        df['updated_weight'] = 0
        df['normalized_weight'] = 0

        for estimator in self.estimators:
            estimator.classifier.fit(df.iloc[: , :n_features].to_numpy(), df['y'].to_numpy())
            df['predicted_y'] = df.apply(lambda row, estimator: int(estimator.classifier.predict(row[:n_features].to_numpy())), args=(estimator,), axis=1)
            df['is_incorrect'] = df['y'] != df['predicted_y']
            
            err = self.calculate_error(df)
            try:
                estimator.alpha = np.log((1 - err) / err)
            except ZeroDivisionError:
                estimator.alpha = 1.5
            self.alpha_sum += estimator.alpha

            stump_performance = err / 2
            df['updated_weight'] = df.apply(lambda row, stump_performance: row['weight'] * np.exp(stump_performance) 
                                        if row['is_incorrect'] else row['weight'] * np.exp(-stump_performance),
                                        args=(stump_performance,), axis=1)

            updated_weights_sum = df['updated_weight'].sum()
            df['normalized_weight'] = df['updated_weight'] / updated_weights_sum

            df = df.sample(n_samples, weights=df['normalized_weight'], ignore_index=True, replace=True)

    def calculate_error(self, df):
        err_weights_sum = (df['weight'] * df['is_incorrect']).sum()
        weights_sum = df['weight'].sum()
        return float(err_weights_sum / weights_sum)

    def predict_proba(self, X):
        probas = []
        for x in X:
            prediction = self.single_predict(x)
            proba = float((prediction + self.alpha_sum) / (2 * self.alpha_sum))
            probas.append([1 - proba, proba])
        return np.array(probas)

    def single_predict(self, x):
        prediction = 0
        for estimator in self.estimators:
            prediction += estimator.alpha * estimator.classifier.predict(x)
        return prediction

    def predict(self, X):
        predictions = [self.single_predict(x) for x in X]
        labels = [1 if prediction > 0 else -1 for prediction in predictions]
        return np.array(labels)

    def score(self, X, y):
        score_sum = 0
        for xi, yi in zip(X, y):
            predicted_label = 1 if self.single_predict(xi) > 0 else -1
            if predicted_label == yi:
                score_sum += 1
        return float(score_sum / len(y))

    def get_params(self):
        return {
            'classifier_cls': self.classifier_cls,
            'params': self.params,
            'n_estimators': self.n_estimators,
        }

    def __repr__(self):
        return f'AdaBoost(classifier_cls={self.classifier_cls}, params={self.params}, n_estimators={self.n_estimators}'

    def __str__(self):
        return 'AdaBoost'
    