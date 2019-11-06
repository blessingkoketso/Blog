import xgboost as xgb
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.datasets import dump_svmlight_file
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold

# Importing the training data 
train_values = pd.read_csv('train_values.csv')


train_values = train_values[['property_type',
'loan_purpose',
'lender',
'loan_amount',
'preapproval',
'state_code',
'county_code',
'applicant_ethnicity',
'applicant_race',
'applicant_sex',
'applicant_income',
'population',
'minority_population_pct',
'ffiecmedian_family_income',
'tract_to_msa_md_income_pct'
 ]]


#We remove or drop the column co_applicant
#train_values = train_values.drop('co_applicant',axis = 1)

# Removing missing values by interpolating the data 
train_values = train_values.interpolate(axis = 1)

# Isolating values and removing the column headers
train_values = train_values.values

# Importing the test data
test_values = pd.read_csv('test_values.csv')

# Selecting 
test_values = test_values[['property_type',
'loan_purpose',
'lender',
'loan_amount',
'preapproval',
'state_code',
'county_code',
'applicant_ethnicity',
'applicant_race',
'applicant_sex',
'applicant_income',
'population',
'minority_population_pct',
'ffiecmedian_family_income',
'tract_to_msa_md_income_pct'
 ]]   


#We remove or drop the column co_applicant
#test_values = test_values.drop('co_applicant',axis = 1)

# Removing missing values by interpolating the data 
test_values = test_values.interpolate(axis = 1)

# Isolating values and removing the column headers
test_values = test_values.values

# Removing missing values by interpolating the data 
#test_values= test_values[['row_id','lender', 'loan_amount', 'loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval','applicant_income', 'applicant_ethnicity', 'applicant_race', 'applicant_sex', 'msa_md' ]]

#test_values = test_values.interpolate(axis = 1)
# We remove or drop the column co_applicant
#test_values = test_values.drop('co_applicant',axis = 1)

# We transform the training data from a dataframe into an array
train_labels = pd.read_csv('train_labels.csv')

# We index the second column of the labels
Labels = np.array(train_labels)
Labels = Labels[:,1]
train_labels = Labels
#Labels = Labels.reshape(Labels.shape[0],)

# display the dimension of the labels
print(train_labels.shape)

# Display the dimensions of the labels
print(train_values.shape)
print(test_values.shape)

# We defining the features
Features = train_values
test_values = test_values
#Labels = np.array(pd.read_csv('C:/Users/bmagabane001/Documents/BKM analytics/DataHack4FI/Capstone/train_values.csv'))
#print(Labels[:,1].shape)
#print(Features.shape)
#print(Labels.shape)
#print(test_values.shape)

x_train,x_test,y_train,y_test = train_test_split(Features,Labels,random_state =1115,test_size = 0.2,train_size = 0.8)

# RandomForestClassifier

rf_clf = RandomForestClassifier(class_weight = "balanced")

## Perform the grid search over the parameters
nr.seed(4455)
rf_clf.fit(Features, Labels)
print(rf_clf.best_estimator_.max_features)
print(rf_clf.best_estimator_.min_samples_leaf)

#xg_cl = xgb.XGBClassifier(objective = 'binary :logistic')

 # Defining the xgboost algorithm
xg_cl = xgb.XGBClassifier()

# Fit model 
xg_cl.fit(x_train,y_train)

# Inquirying on feature importance
Feature_importance = xg_cl.feature_importances_

xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.09,
       max_delta_step=0, max_depth=8, min_child_weight=1, missing=None,
       n_estimators=200, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)



xg_cl.fit(x_train,y_train)

y_pred = xg_cl.predict(x_test)

print(accuracy_score(y_test,y_pred))

######

xg_cl = xgb.XGBClassifier()
learning_rate =[0.01,0.1,0.2,0.3]
max_depth = [2,4,6,8]
n_estimators = [100,150,200]
param_grid = dict(learning_rate=learning_rate,n_estimators = n_estimators,max_depth = max_depth)
kfold = StratifiedKFold(n_splits = 100,shuffle = True,random_state = 7)
grid_search = GridSearchCV(xg_cl,param_grid,scoring = "neg_log_loss",n_jobs=-1,cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
y_pred = grid_search.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

# Neural Network
 x_train, x_test, y_train, y_test = data_processor()
clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100,))
print("Score: ", clf.score(x_test, y_test))


xg_cl.score(x_train,y_train)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


dump_svmlight_file(x_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(x_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations


bst = xgb.train(param, dtrain, num_round)

bst.dump_model('dump.raw.txt')


preds = bst.predict(dtest)

best_preds = np.asarray([np.argmax(line) for line in preds])

print accuracy_score(y_test, best_preds, average='macro')







