
# Improving the accuracy of a model using XGBoost.

There are different ways to improve the accuracy of a model. But in this discussion we are going to focus on gradient based methods. Gradient techniques are ensemble methods. But we are particularly interested in XGBoost. The algorithm is known for winning competitions for structured data. 

## What is XGBoost ?

XGBoost is a supervised machine learning algorithm that uses gradient boosting (GBM) framework. The boosting framework is defined as a numerical optimisation problem where the objective is to minimise the loss of the model. XGBoost belongs to a family of boosting algorithms. The term 'XGBoost' stands for "eXtreme Gradient Boosting". It was developed by Tianqi Chen to specifically improve computational speed and model performance. The algorithm it is an optimized distributed gradient boosting library so it works well on distributed systems.

The XGBoost algorithm is gaining popularity for being better than most machine learning algorithms. It is a “state-of-the-art” machine learning algorithm to deal with structured data. The key to XGBoost high accuracy is the use of boosting. What boosting does best is modifying weak learners into strong ones to improve the prediction accuracy. 

## Comparing Logistic Regression, Random Forests and XGBoost :

In this section, we will compare the performance of a Logistic Regression, Random Forest and XGBoost model.  We will look at an example to test the performance of three models using a mortgage approval dataset. The problem we are trying to solve is the prediction of approvals.

Background on the dataset is adapted from the Federal Financial Institutions Examination Council's (FFIEC). 

### Logistic Regression :
We predict whether an applicant will be granted a loan using logistic regression, 

### Random Forest :  

### XGBoost
The advantage of XGBoost over traditional methods is seen in the execution speed and model performance. It produces superior results due to its software architecture and compatibility to hardware. 

To optimise the XGBoost set the following parameters,  

For max_depth the best value is between 3 and 8 this is the number of trees. The learning_rate should be less than or equal to 0.1 anything learning rate that is small will compromise the prediction.  For n_estimators the best number of estimators to use is between 50 and 150. 

Why use XGBoost

The two reasons to use XGBoost are 
Execution speed.
Model Performance.


It is said that, when you are in doubt use XGBoost.




Introduction to Gradient Boosting
Gradient boosting is one of the most powerful techniques for building predictive models. 
A loss function to be optimized – cross entropy for classification
A weak learner to make prediction – greedily constructed decision tree.
An additive model – used to make weak learners to minimize the loss function
XGBoost is a powerful predictive modeling algorithm, perhaps more powerful than random forest.

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.   The care engineering of the implementation, including:
Parallelisation – tree construction using all of your CPU cores during training.
Distributed Computing – for training very large models using a cluster of machines.
Out-of-Core Computing – for very datasets that don’t fit into memory
Cache Optimisation – data structures and algorithm to make the best use if hardware.

Monitor Performance and Early Stopping







