# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:36:31 2020

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as linear
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import Ridge as ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# The skiprows argument is because the first row is dataset number
literature_data = pd.read_csv('Feed In Data.csv', skiprows=1)

# Extract dataset 3 from the complete dataset frame
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])

# Separate the dataset into training and test sets
literature_data = literature_data.dropna()
litfeat = literature_data.drop(columns = ['exponent'])
litprop = literature_data['exponent']
X_train, X_test, y_train, y_test = train_test_split(litfeat, litprop, test_size=0.2, random_state=5)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform (X_test)

#------------------------LINEAR REGRESSION------------------------------------

"""
gridsearch = GridSearchCV(estimator=linear(),
                          param_grid={
                              'normalize':['True','False']
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Linear Regression MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Linear Regression RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

"""
#------------------------LASSO REGRESSION GRIDSEARCH--------------------------

"""
gridsearch = GridSearchCV(estimator=lasso(random_state=5),
                          param_grid={
                              'alpha':[0.0001,0.001,0.1,1,10,100,1000],
                              'normalize':['True','False']
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Lasso MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Lasso RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

"""
#--------------------------RIDGE REGRESSION GRIDSEARCH------------------------

"""
gridsearch = GridSearchCV(estimator=ridge(random_state=5),
                          param_grid={
                              'alpha':[0.0001,0.001,0.1,1,10,100,1000],
                              'normalize':['True','False']
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Ridge MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Ridge RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

"""
#-------------------------DECISION TREE GRIDSEARCH----------------------------

"""
gridsearch = GridSearchCV(estimator=dtr(random_state=5),
                          param_grid={
                              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                              'min_samples_split':[2,3,4,5],
                              'min_samples_leaf':[1,2,3,4,5]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Decision Tree MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Decision Tree RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

"""
#--------------------------RANDOM FOREST GRIDSEARCH---------------------------

"""
gridsearch = GridSearchCV(estimator=rfr(random_state=5),
                          param_grid={
                              'n_estimators':[20,25,30],
                              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                              'min_samples_split':[2,3,4,5],
                              'min_samples_leaf':[1,2,3,4,5]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Random Forest MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Random Forest RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

"""
#--------------------------SVR GRIDSEARCH-------------------------------------

"""
gridsearch = GridSearchCV(estimator=SVR(kernel='rbf'),
                          param_grid={
                              'gamma':['scale','auto',0.1,1],
                              'C':[3,3.5,4],
                              'epsilon':[0.01,0.1,1]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('SVR MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('SVR RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

"""