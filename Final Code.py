# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 11:36:31 2020

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import Ridge as ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

#This first section splits each dataset into a training and testing set in order
#to initially measure the accuracy of each model. The MAE and RMSE are measured
#for each model by comparing each model's predicted values of conductivity
#against the actual conductivity values.

# Add skiprows b/c it was taking data set names to be column titles
literature_data = pd.read_csv('Feed In Data.csv', skiprows=1)

#Dataset 1 - this line extracts dataset 1 from the complete dataset file
#literature_data = literature_data.drop(columns=['1author3', '1author2', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 2 - this line extracts dataset 2 from the complete dataset file
#literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt', 'temp C', 'exponent', 'Unnamed: 9', '1author'])

#Dataset 3 - this line extracts dataset 3 from the complete dataset file
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])


#separates the dataset into a training and testing set
literature_data = literature_data.dropna()
litfeat = literature_data.drop(columns = ['exponent'])
litprop = literature_data['exponent']
X_train, X_test, y_train, y_test = train_test_split(litfeat, litprop, test_size=0.2, random_state=4)

#-------------------------------linear regression train and test------------------------------------------------------------------

linreg = lr(normalize=True)
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)

linreg_rmse = mean_squared_error(y_test, linreg_pred)
print('linreg MAE: ' + str(sum(abs(linreg_pred - y_test))/(len(y_test))))
print('linreg RMSE: ' + str(np.sqrt(linreg_rmse)))


#-------------------------------lasso regression train and test------------------------------------------------------------------

lassoreg = lasso(normalize=True)
lassoreg.fit(X_train, y_train)
lassoreg_pred = lassoreg.predict(X_test)

lassoreg_rmse = mean_squared_error(y_test, lassoreg_pred)
print('lassoreg MAE: ' + str(sum(abs(lassoreg_pred - y_test))/(len(y_test))))
print('lassoreg RMSE: ' + str(np.sqrt(lassoreg_rmse)))


#-------------------------------ridge regression train and test------------------------------------------------------------------

ridgereg = ridge(normalize=True)
ridgereg.fit(X_train, y_train)
ridgereg_pred = ridgereg.predict(X_test)

ridgereg_rmse = mean_squared_error(y_test, ridgereg_pred)
print('ridgereg MAE: ' + str(sum(abs(ridgereg_pred - y_test))/(len(y_test))))
print('ridgereg RMSE: ' + str(np.sqrt(ridgereg_rmse)))


#-------------------------------decision tree train and test------------------------------------------------------------------

dectree = dtr()
dectree.fit(X_train,y_train)
dectree_pred = dectree.predict(X_test)

dectree_rmse = mean_squared_error(y_test, dectree_pred)
print('dectree MAE: ' + str(sum(abs(dectree_pred - y_test))/(len(y_test))))
print('dectree RMSE: ' + str(np.sqrt(dectree_rmse)))

#-------------------------------random forest train and test---------------------------------------------------------------------

randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)
rf_pred = randomforestmodel.predict(X_test)

rf_rmse = mean_squared_error(y_test, rf_pred)
print('rf MAE: ' + str(sum(abs(rf_pred - y_test))/(len(y_test))))
print('rf RMSE: ' + str(np.sqrt(rf_rmse)))

print('rf percent error: ' + str(sum(((abs(rf_pred - y_test)/y_test)*100))/(len(y_test))))

#plots residual for RF error
residuals = y_test-rf_pred
sns.distplot(residuals, bins=30)
plt.show()

#calculates feature importances that will be used in the custom kernel for SVM 3
importances = randomforestmodel.feature_importances_





#-------------------------------add gridsearch to the random forest (n_estimators, max_depth)----------------------------------
gridsearch = GridSearchCV(estimator=rfr(),
                          param_grid={
                              'n_estimators':[100,200,300,400,500],
                              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                              },
                          scoring='neg_mean_squared_error')

grid_result = gridsearch.fit(X_train, y_train)
best_params = grid_result.best_params_
print(best_params)



#-------------------------------support vector machine 1 train and test------------------------------------------------------------

svr = svm.SVR()
svr = svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

svr_rmse = mean_squared_error(y_test, svr_pred)
print('svr MAE: ' + str(sum(abs(svr_pred - y_test))/(len(y_test))))
print('svr RMSE: ' + str(np.sqrt(svr_rmse)))

#-------------------------------support vector machine 2 train and test----------------------

#creates unweighted custom kernel for SVM 2
def unweighted_kernel(X, Y):
    #return np.dot(np.exp((-1)/(X)), Y.T)
    return np.dot(X, Y.T)

svr2 = svm.SVR(kernel = unweighted_kernel)
svr2 = svr2.fit(X_train, y_train)
svr2_pred = svr2.predict(X_test)

svr2_rmse = mean_squared_error(y_test, svr2_pred)
print('svr2 MAE: ' + str(sum(abs(svr2_pred - y_test))/(len(y_test))))
print('svr2 RMSE: ' + str(np.sqrt(svr2_rmse)))

#-------------------------------support vector machine 3 train and test-----------------------

#creates linear kernel weighted by feature importances (given from RF model) for SVM 3
def importances_kernel(X_1, X_2):
    M = np.array([[importances[0], 0], [0, importances[1]]])
    return np.dot(np.dot(X_1, M), X_2.T)

svr3 = svm.SVR(kernel = importances_kernel)
svr3 = svr3.fit(X_train, y_train)
svr3_pred = svr3.predict(X_test)

svr3_rmse = mean_squared_error(y_test, svr3_pred)
print('svr3 MAE: ' + str(sum(abs(svr3_pred - y_test))/(len(y_test))))
print('svr3 RMSE: ' + str(np.sqrt(svr3_rmse)))


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

#This section trains only the random forest model with the entire dataset 3
#(which was determined to be the most accurate model in Phase 2), without
#splitting it into a training and testing set. The RF model then makes
#predictions based off of the parameters input into the predicted results csv.

'''
#### Note that predicted results.csv is not the GitHub repository!
literature_data = pd.read_csv('Feed In Data.csv')
#testvalues = pd.read_csv('predicted results.csv')

#Dataset 3 - this line extracts dataset 3 from the complete dataset file
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])
#    only dataset 3 was used in this section because it resulted
#    in the most accurate predictions from all models

literature_data = literature_data.dropna()
X_train = literature_data.drop(columns = ['exponent'])
y_train = literature_data['exponent']

#------------------------------Random Forest------------------------------------------------------------
randomforestmodel = rfr()
randomforestmodel.fit(X_train, y_train)

rf_pred = randomforestmodel.predict(testvalues)
'''