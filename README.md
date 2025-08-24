🏦 Bank Customer Churn Prediction
📌 Project Overview

Customer retention is one of the biggest challenges in the banking industry. Churn (or attrition) refers to customers leaving the bank and discontinuing their services. Even a small increase in churn rate can lead to significant revenue loss and negatively impact brand reputation.

This project focuses on building a machine learning model that can predict whether a bank’s customer is likely to exit or not, based on their profile and past behavior. By anticipating churn, banks can take proactive measures to retain valuable clients.

📊 Dataset Information

We use the Churn Modelling dataset available on Kaggle. It contains 10,000 customer records and 14 attributes.

Data Dictionary

| Feature         | Description                                                 |
| --------------- | ----------------------------------------------------------- |
| RowNumber       | Row index (not useful for modeling)                         |
| CustomerId      | Unique identifier for each customer                         |
| Surname         | Customer’s last name                                        |
| CreditScore     | Creditworthiness of the customer                            |
| Geography       | Country of residence                                        |
| City\_Category  | City type (A, B, or C)                                      |
| Gender          | Male or Female                                              |
| Age             | Customer’s age                                              |
| Tenure          | Number of years as a bank client                            |
| Balance         | Account balance                                             |
| NumOfProducts   | Number of bank products used                                |
| HasCrCard       | 1 = owns a credit card, 0 = does not                        |
| IsActiveMember  | 1 = active customer, 0 = inactive                           |
| EstimatedSalary | Customer’s estimated income                                 |
| **Exited**      | **Target variable: 1 = customer left, 0 = customer stayed** |


🔄 Project Workflow

The problem is tackled using the following structured approach:

1. Data Preprocessing

Handle missing values (if any)

Encode categorical variables (e.g., Gender, Geography)

Scale numerical features for better model performance

2. Train-Test Split

70% training data

30% testing data

3. Feature Engineering

Create meaningful transformations (e.g., interaction features, scaling)

Select important predictors

4. Model Building

Train multiple ML algorithms:

Logistic Regression

Random Forest

Gradient Boosting

Neural Networks

Compare performance across models

5. Model Evaluation

Accuracy on training vs. testing data

Confusion matrix, precision, recall, F1-score, ROC-AUC curve

Check for overfitting or underfitting

🎯 Objectives

Predict the probability of a customer leaving the bank.

Help the bank identify at-risk customers early.

Provide insights into which features influence churn the most.

🛠️ Tech Stack

Python: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

Jupyter Notebook for experimentation and analysis

Machine Learning Models: Logistic Regression, Decision Trees, Random Forests, XGBoost, Neural Networks

📌 Key Insights (to be added after training)

Factors strongly influencing churn (e.g., Age, Tenure, Activity Status).

Model performance comparison and selection.

Final recommendation for improving retention strategy.
