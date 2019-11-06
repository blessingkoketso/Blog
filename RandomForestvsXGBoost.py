from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import xgboost as xgb


# Importing the training data 
train_values = pd.read_csv('train_values.csv')

train_values = train_values[['row_id','lender', 'loan_amount', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval','applicant_income', 'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'msa_md' ]]

# Removing missing values by interpolating the data 
train_values = train_values.interpolate(axis = 1)

# We remove or drop the column co_applicant
#train_values = train_values.drop('co_applicant',axis = 1)

# Importing the test data
test_values = pd.read_csv('test_values.csv')

# Removing missing values by interpolating the data 
test_values= test_values[['row_id','lender', 'loan_amount', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval','applicant_income', 'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'msa_md' ]]

test_values = test_values.interpolate(axis = 1)
# We remove or drop the column co_applicant
#test_values = test_values.drop('co_applicant',axis = 1)

# We transform the training data from a dataframe into an array
train_labels =np.array(pd.read_csv('train_labels.csv'))

# We index the second column of the labels
train_labels = train_labels[:,1]

# display the dimension of the labels
print(train_labels.shape)

# Display the dimensions of the labels
print(train_labels)

# We defining the features
Features = np.array(train_values)
Labels = np.array(pd.read_csv('train_labels.csv'))
Labels = Labels[:,1]
#print(Labels[:,1].shape)
Labels = Labels.reshape(Labels.shape[0],)
print(Features.shape)
print(Labels.shape)


nr.seed(1332)
inside = ms.KFold(n_splits=1000, shuffle = True)
nr.seed(1332)
outside = ms.KFold(n_splits=1000, shuffle = True)




## Define the dictionary for the grid search and the model object to search on
param_grid = {"max_features": [2, 3, 5, 8, 10,13], "min_samples_leaf":[3, 5, 8, 10,50,100]}
## Define the random forest model
nr.seed(3456)
rf_clf = RandomForestClassifier(class_weight = "balanced") # class_weight = {0:0.33, 1:0.67}) 

## Perform the grid search over the parameters
nr.seed(4455)
rf_clf = ms.GridSearchCV(estimator = rf_clf, param_grid = param_grid, 
                      cv = inside, # Use the inside folds
                      scoring = 'accuracy',
                      return_train_score = True)
rf_clf.fit(Features, Labels)
print(rf_clf.best_estimator_.max_features)
print(rf_clf.best_estimator_.min_samples_leaf)

# XGBClassifier

XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective='binary:logistic',
 booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 
 colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
 base_score=0.5, random_state=0, seed=None, missing=None)

