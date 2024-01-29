import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
os.system("cls")


class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.models = []
        self.alpha = []
        self.sample_weights = None

    def fit(self, x, y):
        n_samples, n_feature = x.shape
        self.sample_weights = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(x, y, sample_weight=self.sample_weights)
            y_predict = model.predict(x)
            error = np.sum(self.sample_weights * (y != y_predict)) / np.sum(self.sample_weights)
            alpha = .5 * np.log((1 - error) / error)
            self.sample_weights *= np.exp(-y * y_predict * alpha)
            self.models.append(model)
            self.alpha.append(alpha)
    
    def predict(self, x):
        weak_pred = [model.predict(x) for model in self.models]
        weak_pred = np.array(weak_pred)
        self.alpha = np.array(self.alpha)
        weighted_pred = self.alpha.dot(weak_pred)
        return np.sign(weighted_pred)
