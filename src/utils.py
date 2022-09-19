#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import inspect
import settings

import pandas as pd
import numpy as np

from src.model import XGBoost, RandomForest

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Grid Search Selected Feature
params_frst = [  {'score': 'neg_mean_squared_error',
                  'train_size': 0.7,
                  'max_depth': range(5, 15, 2),
                  'n_estimators': (50, 100),
                  'max_features': [.1, .2, .3, .4, .5, .6],
                  'min_samples_leaf': [20, 30],
                  'min_samples_split': [20, 30]},

                 {'score': 'accuracy',
                  'train_size': 0.7,
                  'max_depth': range(5, 15, 2),
                  'n_estimators': (50, 100),
                  'max_features': [.1, .2, .3, .4, .5, .6],
                  'min_samples_leaf': [20, 30],
                  'min_samples_split': [20, 30],} ]

params_xgb = [  {'score': 'neg_mean_squared_error',
                 'train_size': 0.7,
                 "learning_rate": (0.05, 0.10, 0.15),
                 "max_depth": [ 3, 4, 5, 6, 8],
                 "gamma":[ 0.1, 0.5, 1, .5, 2],
                 "sumsamples": [0.25, 0.50, 0.75],
                 "n_estimators": [50, 100, 150],
                 "colsample_bytree":[ 0.25, 0.50]},

                {'score': 'accuracy',
                 'train_size': 0.7,
                 "learning_rate": (0.05, 0.10, 0.15),
                 "max_depth": [ 3, 4, 5, 6, 8],
                 "gamma":[ 0.1, 0.5, 1, .5, 2],
                 "sumsamples": [0.25, 0.50, 0.75],
                 "n_estimators": [50, 100, 150],
                 "colsample_bytree":[ 0.25, 0.50]} ]


def data_augmentation(df, label_attr, opt='div'):
    """ Creates new entry in the dataset considering tuples of the old DS, which are confronted and labeled
        accordingly.

    :param label_attr: label of the feature to be compared
    :param opt: function used to compare the target features
    :type label_attr: string
    :type opt: string

    """
    # parse operation applied between rows
    if opt == 'div':
        opt = lambda x, y: x / y if y != 0 else ""
    elif opt == 'sub':
        opt = lambda x, y: x - y
    elif not inspect.isfunction(opt):
        ValueError('Operation for comparison not correctly defined')

    # creating new labels based on comparison
    """ The main idea is to create couples wich will be the result of the confrontation between two entries
    of the data set """
    n_rows, n_columns = df.shape
    __new_df = pd.DataFrame()
    print(n_rows)
    # for l_column in df.columns:
    # for h in range()
        # column =
    for j in range(0, n_rows):
        for k in range(j+1, n_rows):
            __new_row = [opt(x, y)
                         for (x, y) in zip(df.iloc[j].astype(float), df.iloc[k].astype(float))]
            __new_row = dict(zip(df.columns, __new_row))
            __new_row[label_attr + '_cat'] = 1 if df.iloc[j][label_attr] > df.iloc[k][label_attr] else 2
            # print(__new_row)
            # import sys
            # sys.exit()
            __new_df = __new_df.append(__new_row, ignore_index=True)
        # print(len(__new_df))

    return __new_df

def encode_categorical(df, categorical_list):
    """ Encodes the categorical featurs
    :param categorical_list: list of columns with not encoded categorical features
    :type categorical_list: list(string)
    """
    for category in categorical_list:
        df[category] = df[category].astype('category')
        df[category] = df[category].cat.codes + 1
    return df

def save(df, filename):
    """ Saves the dataframe at the current time in a CSV file
    :param filename: saving name of the dataframe
    :type filename: string
    """
    df.to_csv(filename + '.csv', index=False, sep=';')

def treat_nan(df, max_iter=10, random_state=0):
    """ NaN filler """
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state)
    return pd.DataFrame(imp.fit_transform(df.values), index=df.index, columns=df.columns)

def preprocessing():
    """ Preprocessing dataframe Down Syndrome Children """

    path_to_data = './data/Supplementary Table 1.xlsx'
    df = pd.read_excel(path_to_data, engine='openpyxl')

    #PREPROCESSING
    # remove "Not Yet" string from all cells
    df = df.applymap(lambda x: str(x).replace('Not Yet', '0'))

    # remove < sign from all cells                                             
    remove_grt_sng = lambda x: str(x).replace('<', '')                         
    df = df.applymap(remove_grt_sng)                                           
                                                                               
    # replace , in float with .                                                
    replace_com_dot = lambda x: str(x).replace(',', '.')                       
    df = df.applymap(replace_com_dot)                                          
                                                                               
    # replace with np.nan float nan values (str(1e400*0))                      
    replace_nan = lambda x: np.nan if x == str(1e400 * 0) else x               
    df = df.applymap(replace_nan)   
                                 
    # encodes categorical features 
    df = encode_categorical(df, settings.categorical_list)   

    # define target variable 
    df['AE'] = df['A AE G']
    df['AE'] = df['AE'].combine(df['Tot AE W'],
                                lambda x, y: x if y is np.nan else y)

    # drops partial test columns
    df.drop(columns=df.loc[:, 'A AE G':'Subtest AE W DI'].columns, inplace=True)    
    df.drop(columns=['N', '#ID', 'Mental Retardation Severity', 'Lymphocytes (%)', 'QI G+W'], inplace=True)  
      
    # removes not valid values QI G+W 
    df = df[pd.to_numeric(df['AE']) > 0]     

    # drop indexes column 
    df = df.reset_index()    
    df.drop(columns=['index'], inplace=True)

    # handling nan values with imputer 
    df = treat_nan(df)       

    return df

def feature_importance_model(model, X, y, params={}, features=[], score='neg_mean_squared_error', train_size=.7, t_model='forest'):

    if len(features) == 0:
        # Feature Selection with Boruta
        feature_selected = model.feature_selection(X, y, {'score': score})
        features = [f[0] for f in feature_selected] 
        
        print(features)

    X = X[features] 

    if len(params) == 0:
        # Grid Search
        grid_params = params_xgb if t_model == 'xgb' else params_frst
        
        best_params = model.grid_search(X, y, grid_params[0] if score != 'accuracy' else grid_params[1])

        params = best_params
        params['score'] = score
        params['train_size'] = train_size
       
        print(params)
    
    mdl = model(X, y, params)
    mdl.train()
    
    print('==========================')
    print('==> MSE: ' + str(mdl.test()) if score != 'accuracy' else '==> Accuracy: ' + str(mdl.test())) 
    print('==========================')
    print('==> R2": ' + str(mdl.model.score(X, y)))
    print('==========================')
    print()

    return mdl

def effect_mitigation(df, target_var='Age (months)', test_size=.3):
    """ mitigate effect of avariable and removes it from the datast""" 
    X = df[target_var].values.reshape((-1, 1))

    for column in df.columns:
    # creating learning target 
        y = df[column].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=.2,
                                                            shuffle=False)

        # scaling values
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train)
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

        # training predictor 
        predictor = MLPRegressor(hidden_layer_sizes=(8, 8, 8),
                                 max_iter=1000, 
                                 early_stopping=True).fit(X_train, y_train.ravel())


        #y_pred = predictor.predict(X_test)

        # PLOTTING
        #print(mean_squared_error(y_pred, y_test) / mean_squared_error(predictor.predict(X_train), y_train))
        #plt.plot(scaler_X.transform(X), predictor.predict(scaler_X.transform(X)))
        #plt.scatter(scaler_X.transform(X), scaler_y.transform(y), color='orange')
        #plt.show()
        #plt.close()

        # NORMALIZING 
        Z = predictor.predict(X)
        Z = scaler_y.inverse_transform(Z.reshape(-1, 1))
        df[column] = df[column] / Z.ravel()

    return df

