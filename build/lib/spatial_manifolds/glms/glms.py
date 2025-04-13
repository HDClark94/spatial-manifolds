import pickle
from itertools import combinations

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, KFold

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.glms import bases
from spatial_manifolds.util import get_X, in_leaves


class PoissonGLM:
    def __init__(self, X_, y_, neuron):

        # Prepare data
        shuffle = np.random.permutation(len(y_))
        X, y = in_leaves(X_, shuffle), y_[shuffle]
        X_cv, y_cv = in_leaves(X, slice(int(0.9 * len(y_)))), y[: int(0.9 * len(y_))]
        X_test, y_test = in_leaves(X, slice(int(0.9 * len(y_)), None)), y[int(0.9 * len(y_)) :]

        # Store null model
        self.models = {}
        null_model = PoissonRegressor(alpha=0, fit_intercept=False)
        null_model.fit(get_X(X_cv, ""), y_cv)
        self.models[""] = (
            null_model,
            null_model.score(get_X(X_cv, ""), y_cv),
            null_model.score(get_X(X_test, ""), y_test),
        )

        # Perform grid search with cross-validation
        folds = KFold(n_splits=5)
        for model_type in self.get_model_types(X):
            grid_search = GridSearchCV(
                PoissonRegressor(max_iter=10000),
                {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
                cv=folds,
            )
            grid_search.fit(get_X(X_cv, model_type), y_cv)
            self.models[model_type] = (
                grid_search.best_estimator_,
                grid_search.best_score_,
                grid_search.best_estimator_.score(get_X(X_test, model_type), y_test),
            )
        self.classify(X_test, y_test)
        print(
            neuron,
            ": ",
            self.classification,
            "" if self.classification == "/" else self.models[self.classification][-1],
        )

    def test_better(self, model_type_worse, model_type_better, X, y):
        model_better = self.models[model_type_better][0]
        model_worse = self.models[model_type_worse][0]
        deviance_worse = self.deviance(model_worse.predict(get_X(X, model_type_worse)), y)
        deviance_better = self.deviance(model_better.predict(get_X(X, model_type_better)), y)
        p_value = chi2.sf(
            deviance_worse - deviance_better, df=self.df(model_better) - self.df(model_worse)
        )
        return p_value

    def classify(self, X, y, alpha=0.1):
        simplest = max(self.models, key=len)
        while True:
            simpler_models = {
                model_type: self.test_better(model_type, simplest, X, y)
                for model_type in self.models
                if len(model_type) == len(simplest) - 1 and set(model_type).issubset(set(simplest))
            }
            simpler_models = {
                model_type: p_value
                for model_type, p_value in simpler_models.items()
                if p_value > alpha
            }
            if len(simpler_models) == 0:
                break
            else:
                simplest = max(simpler_models, key=simpler_models.get)
        self.classification = simplest if len(simplest) > 0 else "/"

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load(self, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @classmethod
    def deviance(cls, y_pred, y_true):
        return (
            2
            * (
                (y_true * np.log((y_true + np.finfo(float).eps) / y_pred)) - (y_true - y_pred)
            ).sum()
        )

    @classmethod
    def df(cls, model):
        return len(model.coef_) + (1 if model.fit_intercept else 0)

    @classmethod
    def apply_bases(cls, behaviour, task):
        return {
            data_type[0].split("_")[0]: getattr(bases, f"spline_{data_config['dim']}")(
                *[behaviour[data_subtype] for data_subtype in data_type],
                data_config["num_bins"],
                data_config["bounds"],
            )
            for data_type, data_config in get_bin_config(task).items()
        }

    @classmethod
    def get_model_types(cls, X):
        return ["".join(comb) for r in range(1, len(X) + 1) for comb in combinations(X.keys(), r)]
