
# Improving the accuracy of a model using XGBoost.

There are different ways to improve the accuracy of a model. But in this discussion we are going to focus on gradient based methods. Gradient techniques are ensemble methods. But we are particularly interested in XGBoost. The algorithm is known for winning competitions for structured data. 

## What is XGBoost ?

XGBoost is a supervised machine learning algorithm that uses gradient boosting (GBM) framework. The boosting framework is defined as a numerical optimisation problem where the objective is to minimise the loss of the model. XGBoost belongs to a family of boosting algorithms. The term 'XGBoost' stands for "eXtreme Gradient Boosting". It was developed by Tianqi Chen to specifically improve computational speed and model performance. The algorithm it is an optimized distributed gradient boosting library so it works well on distributed systems.

The XGBoost algorithm is gaining popularity for being better than most machine learning algorithms. It is a “state-of-the-art” machine learning algorithm to deal with structured data. The key to XGBoost high accuracy is the use of boosting. What boosting does best is modifying weak learners into strong ones to improve the prediction accuracy. 

## Comparing Logistic Regression, Random Forests and XGBoost :

In this section, we will compare the performance of a Logistic Regression, Random Forest and XGBoost model.  We will look at an example to test the performance of three models using a mortgage approval dataset. The problem we are trying to solve is the prediction of approvals.

Background on the dataset is adapted from the Federal Financial Institutions Examination Council's (FFIEC). Through feature engineering we selected the following features, 

- property_type
- loan_purpose
- lender
- loan_amount
- preapproval
- state_code
- county_code
- applicant_ethnicity
- applicant_race
- applicant_sex
- applicant_income
- population
- minority_population_pct
- ffiecmedian_family_income
- tract_to_msa_md_income_pct
- co_applicant
 
### Logistic Regression :
We predict whether an applicant will be granted a loan using logistic regression, 

Below is a code snippet of Logistic regression,

- X_train,X_test,y_train,y_test = train_test_split(Features,Labels,random_state =1115,test_size = 0.2,train_size = 0.8)

- lr_clf = LogisticRegression()

- lr_clf.fit(X_train,y_train)

- y_pred = lr_clf.predict(X_test)

- score = accuracy_score(y_test,y_pred)

If you run the code snippet above you will get an accuracy of 0.62007. 

### Random Forest : 
We predict whether an applicant will be granted a loan but now we are using random forests.


Below is a code snippet of Random forest,

- X_train,X_test,y_train,y_test = train_test_split(Features,Labels,random_state =1115,test_size = 0.2,train_size = 0.8)

- rf_clf = RandomForestClassifier(class_weight = "balanced")

- rf_clf.fit(X_train,y_train)

- y_pred = rf_clf.predict(X_test)

- score = accuracy_score(y_test,y_pred)

If you run the code snippet above you will get an accuracy of 0.65767.

### XGBoost :
We predict whether an applicant will be granted a loan but now we are using XGBoost. To get the best out of XGBoost we optimise the parameters in the following, 

- For max_depth the best value is between 3 and 8 this is the number of trees. 
- The learning_rate should be less than or equal to 0.1 anything learning rate that is small will compromise the prediction. 
- For n_estimators the best number of estimators to use is between 50 and 150. 

Below is a code snippet of XGBoost,

- X_train,X_test,y_train,y_test = train_test_split(Features,Labels,random_state =1115,test_size = 0.2,train_size = 0.8)

- xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=8, min_child_weight=1, missing=None,
       n_estimators=150, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)

- xg_cl.fit(X_train,y_train)

- y_pred = xg_cl.predict(X_test)

- print(accuracy_score(y_test,y_pred))

If you run the code snippet above you will get an accuracy of 0.71299.

If we compare the results we can see that the clear winner when it comes to accuracy is XGBoost. The advantage of XGBoost is in the execution speed and model performance. It produces superior results due to the architecture and compability to hardware.


### Why use XGBoost

The two reasons to use XGBoost are 
- Execution speed.
- Model Performance.

It is said that, when you are in doubt use XGBoost.












