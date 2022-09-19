#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import xgboost as xgb

# import graphviz
# from sklearn.inspection import permutation_importance
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from boruta import BorutaPy
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor


class Model():
    def __init__(self, X, y, params):
        self.X = X
        self.y = y
        self.train_size = params['train_size']
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(self.X,
                                                       self.y,
                                                       random_state=42,
                                                       train_size=self.train_size)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        pass

    def importance(self):
        pass

    def plot_importance(self):
        pass

    def grid_search(self):
        pass

    def predict(self, x):
        return self.model.predict(x)

class TreeModel(Model):
    def __init__(self, X, y, params):
        Model.__init__(self, X, y, params)
        self.score = params['score']  # 'accuracy' or 'neg_means_squared_error'
        self.max_depth = params['max_depth']  # maximum depth tree
        self.max_features = params['max_features']  # maximum number of attributes
        self.min_samples_leaf = params['min_samples_leaf']  # minimum number of examples to be considered a leaf
        self.min_samples_split = params['min_samples_split']  # minimum number of examples to a branch

    def importance(self, cv=10):
        """ Evaluates the model

                :param cv: number of folds for the validation
                :type cv: int

                """
        predictions = self.model.predict(self.X_test)

        # cross validation
        self.scores = cross_validate(self.model,
                                     self.X,
                                     self.y,
                                     cv=cv,
                                     scoring=self.score,
                                     return_estimator=True)

        self.mean = np.mean([tree.feature_importances_ for tree in self.scores['estimator']],
                              axis=0)

        self.var = np.var([tree.feature_importances_ for tree in self.scores['estimator']],
                            axis=0)

        self.accuracy = round(100 * (accuracy_score(self.y_test, predictions)), 2) if self.score == 'accuracy' else None

        return pd.Series(self.mean).sort_values(ascending=False)

    def permutation_importance(self):
        return pd.Series(
            permutation_importance(self.model, self.X_test, self.y_test).importances_mean).sort_values(ascending=False)

    def test(self):
        y_pred = self.model.predict(self.X_test)
        scorer = classification_report if self.score == 'accuracy' else mean_squared_error
        return scorer(y_pred, self.y_test)

    def plot_permutation_importance(self, n_features=10):
        perm_importance = self.permutation_importance()[:n_features]
        #sorted_idx = perm_importance.importances_mean.argsort()
        sns.set(font_scale=3.5)

        sns.barplot(x=perm_importance,
                    y=self.X_test.columns.values[perm_importance.index])

        sns.set(rc={'figure.figsize': (30, 20)})
        plt.xlabel('Feature Importance Score', fontsize=40)
        plt.ylabel('Features', fontsize=40)
        plt.title("Visualizing Pemutation Important Features", fontsize=40)
        plt.show()

    def plot_importance(self, n_features=10, title=False):
        """ Plots the most important features

        :param n_features: number of  feature plotted (default 20)
        :type n_features: int

        """
        self.importance()
        features = self.importance()
        n_features = len(features) if n_features is None else n_features
        features = features[:n_features]
        sns.set(font_scale=3.5)
        sns.barplot(x=features, y=self.X.columns.values[features.index])
        sns.set(rc={'figure.figsize': (10, 5)})
        #if title:
        plt.xlabel('Feature Importance Score', fontsize=30)
        plt.ylabel('Features', fontsize=30)
            #plt.title("Visualizing Important Features", fontsize=40)
        plt.show()

class RandomForest(TreeModel):
    def __init__(self, X, y, params):
        TreeModel.__init__(self, X, y, params)

        self.n_estimators = params['n_estimators']  # number of evaluated trees

        # assigning w.r.t the mode the model type: 'cls', 'rgs'
        self.model = RandomForestClassifier if self.score == 'accuracy' else RandomForestRegressor

        # initialization model
        self.model = self.model(max_depth=self.max_depth,
                                max_features=self.max_features,
                                min_samples_leaf=self.min_samples_leaf,
                                min_samples_split=self.min_samples_split,
                                n_estimators=self.n_estimators,
                                random_state=False,
                                verbose=False)

    @classmethod
    def feature_selection(self, X, y, params):
        if params['score'] == 'accuracy':
            model = RandomForestClassifier
        else:
            model = RandomForestRegressor

        model = model(random_state=False, verbose=False)

        selector = BorutaPy(model, n_estimators = 'auto', verbose = 0, random_state = 1)
        selector.fit(X.values, y.values)
        X_filtered = selector.transform(X.values)
        feature_ranks = []
        for i, feature in enumerate(X.columns):
            if selector.support_[i]:
                feature_ranks.append((feature, selector.ranking_[i]))
        return feature_ranks

    @classmethod
    def grid_search(self, X, y, params):

        if params['score'] == 'accuracy':
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        grid = GridSearchCV(model, param_grid={
            'max_depth': params['max_depth'],
            'n_estimators': params['n_estimators'],
            'max_features': params['max_features'],
            'min_samples_leaf': params['min_samples_leaf'],
            'min_samples_split': params['min_samples_split'],
        })

        grid_result = grid.fit(X, y)

        return grid_result.best_params_


class ExtraTree(TreeModel):
    """ Class used to represent the Extra Random Trees model """
    def __init__(self, X, y, params):
        """ Class initialization

        :param X: training observations
        :param y: trainnig target
        :param params: set of paramenter of the model
        :type X: pandas.DataFrame
        :type y: pandas.Series
        :type params: dictionary
        """
        TreeModel.__init__(self, X, y, params)

        self.n_estimators = params['n_estimators']  # number of evaluated trees

        # assigning w.r.t the mode the model type: 'cls', 'rgs'
        self.model = ExtraTreesClassifier if self.score == 'accuracy' else ExtraTreesRegressor

        # initialization model
        self.model = self.model(max_depth=self.max_depth,
                                max_features=self.max_features,
                                min_samples_leaf=self.min_samples_leaf,
                                min_samples_split=self.min_samples_split,
                                n_estimators=self.n_estimators,
                                random_state=False,
                                verbose=False)

    @classmethod
    def feature_selection(self, X, y, params):
        if params['score'] == 'accuracy':
            model = ExtraTreesClassifier
        else:
            model = ExtraTreesRegressor

        model = model(random_state=False, verbose=False)

        selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=1)
        selector.fit(X.values, y.values)
        X_filtered = selector.transform(X.values)
        feature_ranks = []
        for i, feature in enumerate(X.columns):
            if selector.support_[i]:
                feature_ranks.append((feature, selector.ranking_[i]))
        return feature_ranks

    @classmethod
    def grid_search(self, X, y, params):

        if params['score'] == 'accuracy':
            model = ExtraTreesClassifier()
        else:
            model = ExtraTreesRegressor()

        grid = GridSearchCV(model, param_grid={
            'max_depth': params['max_depth'],
            'n_estimators': params['n_estimators'],
            'max_features': params['max_features'],
            'min_samples_leaf': params['min_samples_leaf'],
            'min_samples_split': params['min_samples_split'],
        })

        grid_result = grid.fit(X, y)

        return grid_result.best_params_


class XGBoost(Model):

    def __init__(self, X, y, params):
        Model.__init__(self, X, y, params)
        self.learning_rate = params["learning_rate"]
        self.max_depth = params["max_depth"]
        self.gamma = params["gamma"]
        self.subsamples = params["sumsamples"]
        self.n_estimators = params["n_estimators"]
        self.colsample_bytree = params["colsample_bytree"]
        self.score = params['score']

        self.model = xgb.XGBClassifier if self.score == 'accuracy' else xgb. XGBRegressor

        self.model = self.model(learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                gamma=self.gamma,
                                subsamples=self.subsamples,
                                n_estimators=self.n_estimators,
                                colsample_bytree=self.colsample_bytree)

    def test(self):
        y_pred = self.model.predict(self.X_test)
        scorer = classification_report if self.score == 'accuracy' else mean_squared_error
        return scorer(y_pred, self.y_test)

    @classmethod
    def feature_selection(self, X, y, params):
        if params['score'] == 'accuracy':
            model = xgb.XGBClassifier
        else:
            model = xgb.XGBRegressor

        model = model(random_state=False, verbose=False)

        selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=1)
        selector.fit(X.values, y.values)
        X_filtered = selector.transform(X.values)
        feature_ranks = []
        for i, feature in enumerate(X.columns):
            if selector.support_[i]:
                feature_ranks.append((feature, selector.ranking_[i]))
        return feature_ranks

    @classmethod
    def grid_search(self, X, y, params):

        if params['score'] == 'accuracy':
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor(objective ='reg:linear')

        grid = GridSearchCV(model, param_grid={
            "learning_rate":  params["learning_rate"],
            "max_depth": params["max_depth"],
            "gamma": params["gamma"],
            "sumsamples": params["sumsamples"],
            "n_estimators": params["n_estimators"],
            "colsample_bytree": params["colsample_bytree"],
        })
        print(grid)
        grid_result = grid.fit(X, y)

        return grid_result.best_params_

    def importance(self):
        sort = lambda x : dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
        return sort(self.model.get_booster().get_score())

    def plot_importance(self, n_features=10, title=False):
        """ Plots the most important features

        :param n_features: number of  feature plotted (default 20)
        :type n_features: int

        """
        #self.importance()
        features = self.importance()
        n_features = len(features) if n_features is None else n_features
        #features = features[:n_features]
        labels = list(features.keys())[:n_features]
        values = list(features.values())[:n_features]
        sns.set(font_scale=3.5)
        sns.barplot(x=values, y=labels)
        sns.set(rc={'figure.figsize': (10, 5)})
        #if title:
        plt.xlabel('Feature Importance Score', fontsize=30)
        plt.ylabel('Features', fontsize=30)
            #plt.title("Visualizing Important Features", fontsize=40)
        plt.show()

# class LogisticRegressionModel(Model):
#     def __init__(self, X, y, params):
#
#         Model.__init__(self, X, y, params)
#         self.__penalty = params['penalty']
#         self.__C = params['C']
#         self.__class_weight = params['class_weight']
#         self.__max_iter = params['max_iter']
#         self.__model = LogisticRegression(penalty=self.__penalty,
#                                           C=self.__C,
#                                           solver='saga',
#                                           max_iter=self.__max_iter,
#                                           class_weight=self.__class_weight)
#
#     @classmethod
#     def grid_search(cls, X, y, params):
#         self = cls(X, y, params)
#
#         self.__model = GridSearchCV(LogisticRegression,
#                                     param_grid={
#                                         'C': np.linspace(0, 1, 11),
#                                         'max_iter': (100, 1000, 5000),
#                                     })
#
#         self.__model.fit(self.__X, self.__y)
#         return self
#
#     def importance(self):
#         self.__model._coef[0]
#
#     def plot_importance(self):
#         imp = self.importance()
#         plt.figure(figsize=(30, 10))
#         plt.bar(self.__X.columns, imp)
#         plt.show()

