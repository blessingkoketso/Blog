import xgboost as xgb
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Importing the training data 
train_values = pd.read_csv('train_values.csv')

# Feature selection, headers of interest.
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
'tract_to_msa_md_income_pct',
'co_applicant'
 ]]

# Converting strings to numericals.
le = preprocessing.LabelEncoder()
train_values['co_applicant']=le.fit_transform(train_values['co_applicant'])

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
'tract_to_msa_md_income_pct',
'co_applicant'
 ]]   

# Converting strings to numericals.
le = preprocessing.LabelEncoder()
test_values['co_applicant']=le.fit_transform(test_values['co_applicant'])

# Removing missing values by interpolating the data 
test_values = test_values.interpolate(axis = 1)

# Isolating values and removing the column headers
test_values = test_values.values

# We transform the training data from a dataframe into an array
train_labels = pd.read_csv('train_labels.csv')

# We index the second column of the labels
Labels = np.array(train_labels)
Labels = Labels[:,1]
train_labels = Labels


# display the dimension of the labels
print(train_labels.shape)

# Display the dimensions of the labels
print(train_values.shape)
print(test_values.shape)

# We defining the features
Features = train_values
test_values = test_values

# Splitting the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(Features,Labels,random_state =1115,test_size = 0.2,train_size = 0.8)

# XGBoost
xg_cl = xgb.XGBClassifier()

# Training the model
xg_cl.fit(X_train,y_train)

# Predicting
y_pred = xg_cl.predict(X_test)

# Accuracy
print(accuracy_score(y_test,y_pred))