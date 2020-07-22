# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:36:31 2020

"""

import numpy as np
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import Ridge as ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------

#This first section splits each dataset into a training and testing set in order
#to initially measure the accuracy of each model. The MAE and RMSE are measured
#for each model by comparing each model's predicted values of conductivity
#against the actual conductivity values.

# Add skiprows b/c it was taking data set names to be column titles
literature_data = pd.read_csv('Feed In Data.csv', skiprows=1)

#Dataset 3 - this line extracts dataset 3 from the complete dataset file
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])

#separates the dataset into a training and testing set
literature_data = literature_data.dropna()
litfeat = literature_data.drop(columns = ['exponent'])
litprop = literature_data['exponent']
X_train, X_test, y_train, y_test = train_test_split(litfeat, litprop, test_size=0.2, random_state=4)


#-------------------------------linear regression train and test------------------------------------------------------------------
"""
linreg = lr(normalize=True)
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)

linreg_rmse = mean_squared_error(y_test, linreg_pred)
print('linreg MAE: ' + str(sum(abs(linreg_pred - y_test))/(len(y_test))))
print('linreg RMSE: ' + str(np.sqrt(linreg_rmse)))


#-------------------------------lasso regression gridsearch------------------------------------------------------------------

lassoreg = lasso(normalize=True)
lassoreg.fit(X_train, y_train)
lassoreg_pred = lassoreg.predict(X_test)

lassoreg_rmse = mean_squared_error(y_test, lassoreg_pred)
print('lassoreg MAE: ' + str(sum(abs(lassoreg_pred - y_test))/(len(y_test))))
print('lassoreg RMSE: ' + str(np.sqrt(lassoreg_rmse)))

"""
#-------------------------------ridge regression gridsearch------------------------------------------------------------------

"""
gridsearch = GridSearchCV(estimator=ridge(),
                          param_grid={
                              'alpha':[0.001, 0.01, 0.1, 1],
                              'normalize':['True','False']
                              },
                          scoring='neg_mean_squared_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('grid MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('grid RMSE: ' + str(np.sqrt(grid_rmse)))
best_params = grid_result.best_params_
best_score = grid_result.best_score_
print(best_params)
print(math.sqrt(abs(best_score)))
"""

#-------------------------------add gridsearch to decision trees and random forests ---------------------------------

# begin decision trees
"""
gridsearch = GridSearchCV(estimator=dtr(random_state=4),
                          param_grid={
                              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12],
                              'min_samples_split':[2,3,4,5],
                              'min_samples_leaf':[1,2,3,4,5]
                              },
                          scoring='neg_mean_squared_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('grid MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('grid RMSE: ' + str(np.sqrt(grid_rmse)))
best_params = grid_result.best_params_
best_score = grid_result.best_score_
print(best_params)
print(math.sqrt(abs(best_score)))
"""


# begin random forests

### Need to go lower on n_estimators! Shouldn't come up against a boundary
gridsearch = GridSearchCV(estimator=rfr(random_state=4),
                          param_grid={
                              'n_estimators':[50,75,100,125,150,175,200],
                              'max_depth':[5,6,7,8,9,10,11,12,13,14,15]
                              },
                          scoring='neg_mean_squared_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('grid MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('grid RMSE: ' + str(np.sqrt(grid_rmse)))
best_params = grid_result.best_params_
best_score = grid_result.best_score_
print(best_params)
print(math.sqrt(abs(best_score)))

# Check max_depth against overfitting!
#-------------------------------support vector machine 1 train and test------------------------------------------------------------
"""
gridsearch = GridSearchCV(estimator=SVR(),
                          param_grid={
                              'kernel':['rbf','linear','poly','sigmoid'],
                              'gamma':[0.01, 0.1, 1, 10, 100],
                              'C':[0.01, 0.1, 1, 10, 100]
                              },
                          scoring='neg_mean_squared_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('grid MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('grid RMSE: ' + str(np.sqrt(grid_rmse)))
best_params = grid_result.best_params_
best_score = grid_result.best_score_
print(best_params)
print(math.sqrt(abs(best_score)))
"""


# Cross validation
# Standardization and normalization
# Exhaustive parameter gridsearch for all models
# Get TDS to check paper against the revisions?
# Widen SVR param grid