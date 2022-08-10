from posixpath import split
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
)
from sklearn import model_selection
from sklearn import metrics


class CrossValidation:
    def __init__(self, model, conf):
        """
        Initialize the class
        :param model: model to be used for cross validation
        :param conf: configuration for cross validation. See below for details
            conf = {
                'cv_model': 'KFold', # string: cross validation model to be used (KFold, TimeSeriesSplit, ShuffleSplit)
                'cv_n_splits': 5, # int: number of splits for cross validation
                'cv_test_size': 0.2, # float: test size for cross validation
                'cv_random_state': 42, # int: random state for cross validation
                'cv_shuffle': True, # boolean: shuffle for cross validation
                'split_test_size': 0.2, # float: test size for train_test_split
                'split_random_state': 42, # int: random state for train_test_split
                'split_shuffle': True, # boolean: shuffle for train_test_split
                'scoring': ['neg_mean_squared_error', 'neg_mean_absolute_error'], # list: scoring for cross validation
                'n_jobs': 2, # int: number of jobs for cross validation
                'return_train_score': True, # boolean: return train score for cross validation
                'return_estimator': True, # boolean: return estimator for cross validation
                'metrics': ['mean_absolute_error', 'mean_squared_error'], # list: metrics for cross validation regarding to scoring list names and order in list
                }
        :return: CrossValidation instance
        """
        self.model = model
        self.conf = conf

    def __call__(self, *args, **kwds):
        """
        Call the class
        :param args: arguments for the class call
        :param kwds: keyword arguments for the class call
        :return: cross validation results
        """
        return self.cross_validation(*args, **kwds)

    def cross_validation(self, X, y):
        """
        Cross validation with cross_validate
        :param X: features
        :param y: target
        :return: cross validation results
        """
        conf = self.conf
        cv_model = getattr(model_selection, conf["cv_model"])
        conf.pop("cv_model")
        split_conf = {k.replace("split_", ""): v for k, v in conf.items() if k.startswith("split_")}
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            **split_conf
        )

        model_metrics = conf["metrics"]
        cv_conf = {k.replace("cv_", ""): v for k, v in conf.items() if k.startswith("cv_")}
        cv = cv_model(**cv_conf)
        
        # :TODO why model needs to be fit bofer cross_validate?
        self.model.fit(X_train, y_train)
        cv_results = cross_validate(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring=conf["scoring"],
            return_train_score=conf["return_train_score"],
            n_jobs=conf["n_jobs"],
        )
        cv_results = pd.DataFrame(cv_results)
        cv_results.pop("fit_time")
        cv_results.pop("score_time")
        test_scores = cv_results
        ytrain = self.model.predict(X_train)

        train_model_metrics_result = []
        for metric in model_metrics:
            m = getattr(metrics, metric)
            train_model_metrics_result.append({metric: m(y_train, ytrain)})

        test_model_metrics_result = []
        ypred = self.model.predict(X_test)
        for metric in model_metrics:
            m = getattr(metrics, metric)
            test_model_metrics_result.append({metric: m(y_test, ypred)})
        cv_results = cv_results.append(np.mean(test_scores, axis=0), ignore_index=True)
        cv_results = cv_results.append(np.std(test_scores, axis=0), ignore_index=True)
        num_slplit = conf["cv_n_splits"]
        for i in range(num_slplit):
            cv_results = cv_results.rename(index={i: f"slice {i}"})

        cv_results = cv_results.rename(
            index={len(cv_results) - 2: "mean", len(cv_results) - 1: "std"}
        )
        
        train_model_metrics_result = [list(x.values())[0] for x in train_model_metrics_result]
        test_model_metrics_result = [list(x.values())[0] for x in test_model_metrics_result]
        
        final = [item for sublist in zip(test_model_metrics_result, train_model_metrics_result) for item in sublist]
        
        cv_results.loc[len(cv_results)] = final
        cv_results = cv_results.rename(index={len(cv_results) - 1: "final"})

        return cv_results
