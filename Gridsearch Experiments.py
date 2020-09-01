# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 11:32:15 2020

@author: xf290
"""

import math
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
import seaborn as sns
import matplotlib.pyplot as plt

# The skiprows argument is because the first row is dataset number
literature_data = pd.read_csv('Feed In Data.csv', skiprows=1)

#Dataset 1 - this line extracts dataset 1 from the complete dataset file
literature_data = literature_data.drop(columns=['1author3', '1author2', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 2 - this line extracts dataset 2 from the complete dataset file
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 3 - this line extracts dataset 3 from the complete dataset file
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])


# Separate the dataset into training and test sets
literature_data = literature_data.dropna()
litfeat = literature_data.drop(columns = ['exponent'])
litprop = literature_data['exponent']
X_train, X_test, y_train, y_test = train_test_split(litfeat, litprop, test_size=0.2, random_state=4)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform (X_test)

#------------------------LINEAR REGRESSION GRIDSEARCH------------------------------------
""" the only parameter we input into the gridsearch is 'normalize', which was optimized
to be 'True' """

gridsearch = GridSearchCV(estimator=linear(), cv=5,
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


#------------------------LASSO REGRESSION GRIDSEARCH--------------------------
""" the best values for each parameter came out as: 'alpha'=0.001 and
'normalize'=True """

gridsearch = GridSearchCV(estimator=lasso(random_state=4), cv=5,
                          param_grid={
                              'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000],
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


#--------------------------RIDGE REGRESSION GRIDSEARCH------------------------
""" the best values for each parameter came out as: 'alpha'=0.001 and
'normalize'=True """

gridsearch = GridSearchCV(estimator=ridge(random_state=4), cv=5,
                          param_grid={
                              'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000],
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


#-------------------------DECISION TREE GRIDSEARCH----------------------------
""" the best values for each parameter came out as: 'max_depth'=12,
'min_samples_leaf'=1, and 'min_samples_split'=2 """

gridsearch = GridSearchCV(estimator=dtr(random_state=4), cv=5,
                          param_grid={
                              'max_depth':[10,20,30,40,50],
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


#--------------------------RANDOM FOREST GRIDSEARCH---------------------------
""" the best values for each parameter came out as: 'max_depth'=14,
'min_samples_leaf'=1, 'min_samples_split'=2, and 'n_estimators'=12 """

gridsearch = GridSearchCV(estimator=rfr(random_state=4), cv=5,
                          param_grid={
                              'n_estimators':[10,20,40,70,100,500,1000],
 """we selected the n_estimators at approximately the point of diminishing
 performance returns, which is why the optimal value ended up being so low"""
                              'max_depth':[10,20,30,40,50],
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

residuals = y_test-grid_pred
sns.distplot(residuals, bins=40)
plt.show()

#--------------------------SVR GRIDSEARCH-------------------------------------
""" the best values for each parameter came out as: 'C'=4, 'epsilon'=0.1,
and 'gamma'=scale """

gridsearch = GridSearchCV(estimator=SVR(kernel='rbf'), cv=5,
                          param_grid={
                              'gamma':['scale','auto',0.1,1],
                              'C':[5,10,15,20,25,30],
                              'epsilon':[0.0001,0.001,0.01,0.1,1,10,100,1000]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('SVR MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('SVR RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

