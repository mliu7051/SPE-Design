#This section trains only the random forest model with the entire dataset 3
#(which was determined to be the most accurate model in Phase 2), without
#splitting it into a training and testing set. The RF model then makes
#predictions based off of the parameters input into the predicted results csv.

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
