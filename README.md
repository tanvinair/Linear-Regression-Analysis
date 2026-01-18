# Linear-Regression-Analysis
 The objective of this project is to analyze the dataset, perform preprocessing and  exploratory data analysis (EDA), and then build two different models for regression:   1. Stochastic Gradient Descent using the SGDRegressor library of Scikit-learn.  2. Ordinary Linear Regression using the statsmodels library.

 Assignment 1 README File: Linear Regression Analysis 
Created by: Shraddha Gangaram and Tanvi Nair 
This assignment implements linear regression using two approaches: 
1. Scikit-learn SGDRegressor: A library implementation of linear regression using 
stochastic gradient descent with hyperparameter tuning. 
2. OLS Regression: Ordinary Least Squares regression using the statsmodels library to 
obtain detailed statistical insights. 
Assignment Objective: The objective is to explore the dataset, select important features, train 
and evaluate models, and interpret the results. 
Dataset Choice: Both approaches use the Red Wine Quality dataset from the UCI Machine 
Learning Repository (Link: https://archive.ics.uci.edu/dataset/186/wine+quality) 
Files Attached: 
● cs_4372_assignment_1.py – Full code implementation for SGDRegressor and OLS 
● sgd_results.csv – Hyperparameter tuning results for SGDRegressor 
● CS 4372 - Assignment 1 Report Analysis.pdf – Report with analysis, interpretations, and 
plots 
● CS 4372 - Assignment 1 README.pdf – Project instructions and overview 
● CS4372_CoverPage_Assignment1.pdf – UTD cover sheet with student details and 
references 
Required Python Packages: 
● pandas 
● numpy 
● matplotlib 
● seaborn 
● scikit-learn 
● statsmodels 
Installation: 
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels 
Assumptions 
● The problem is treated as a regression. 
● Each wine observation is independent. 
● Linearity is assumed between selected features and wine quality. 
How to Run SGDRegressor Model - Running the notebook/script will: 
1. Load the dataset directly from the UCI repository. 
2. Generate visualizations to explore the data: 
○ Histograms for all features 
○ Correlation heatmap 
○ Correlation bar plot highlighting each predictor’s correlation with wine quality 
3. Split the dataset into training (80%) and testing (20%) sets. 
4. Standardize and normalize features to ensure comparability across variables. 
5. Train an SGDRegressor model with hyperparameter tuning over: 
○ loss functions 
○ penalty types 
○ alpha values 
○ learning_rate options  
○ Initial step sizes eta0 
6. Output evaluation metrics: 
○ Train/Test R² 
○ Mean Squared Error (MSE) 
○ Mean Absolute Error (MAE) 
○ Explained Variance Score 
7. Print the model coefficients for interpretation. 
8. Multiple combinations of loss, penalty, alpha, learning rate, and eta0 were systematically 
explored. The best-performing set is reported, and the full tuning log is saved in 
sgd_results.csv. 
Key results from the SGDRegressor Model: 
● Selected Important Features: ['alcohol', 'volatile acidity', 'sulphates', 'citric acid'] 
● Best Hyperparameters: 
○ Loss: squared_error 
○ Penalty: l1 
○ Alpha: 0.001 
○ Learning rate: constant 
○ Eta0: 0.01 
● Model Coefficients: 
○ Alcohol: 0.3245 (positive impact) 
○ Sulphates: 0.0997 (slight positive impact) 
○ Citric acid: 0.0000 (no impact) 
○ Volatile acidity: -0.2503 (negative impact) 
● Evaluation Metrics: 
○ Train/Test R²: 0.422 
○ Mean Squared Error (MSE): 0.409 
○ Mean Absolute Error (MAE): 0.495 
○ Explained Variance Score: 0.422 
● Conclusion: The tuned SGD model indicates that alcohol and sulphates increase 
predicted wine quality, while volatile acidity decreases it, and citric acid contributes 
minimally once the other predictors are included. Overall, the model explains a moderate 
portion of the variation, with a test R² of 0.422, which aligns with the findings of the OLS 
model that follows, while achieving slightly stronger predictive performance. 
How to Run OLS Regression - Running the notebook/script will: 
1. Load the dataset (preprocessed with the same selected features as the SGDRegressor 
Model above) 
2. Add a constant term to the predictors for the intercept. 
3. Fit the OLS regression model using statsmodels. 
4. Output model summary and diagnostics, including: 
○ R-squared and adjusted R-squared 
○ F-statistic and p-values for overall model significance 
○ Coefficients and p-values for each predictor 
○ Diagnostic tests (Omnibus, Jarque-Bera, Skew, Kurtosis, Durbin-Watson, 
Condition Number) 
5. Interpret results: Identify statistically significant predictors and their effects on wine 
quality. 
6. In addition to coefficients and p-values, residuals were examined using Omnibus, 
Jarque-Bera, Skew, Kurtosis, Durbin-Watson, and Condition Number tests to ensure 
model validity and detect potential multicollinearity or non-normality. 
Key results from the OLS Regression Model:  
● Selected Important Features: ['alcohol', 'volatile acidity', 'sulphates', 'citric acid'] 
● Key Coefficients and P-values: 
○ Constant: 2.6629 
○ Alcohol: 0.3077 (p < 0.05) 
○ Sulphates: 0.6352 (p < 0.05) 
○ Citric acid: -0.0616 (p = 0.633) – not significant 
○ Volatile acidity: -1.2120 (p < 0.05) 
● R² and Adjusted R²: 
○ R²: 0.322 
○ Adjusted R²: 0.319 
● Residual & Model Diagnostics: 
○ Omnibus: 18.295 (Prob = 0.000) 
○ Jarque-Bera (JB): 27.534 (Prob = 1.05e-06) 
○ Skew: -0.150 
○ Kurtosis: 3.720 
○ Durbin-Watson: 1.952 
○ Condition Number: 137 
● Conclusion: The OLS model finds that alcohol and sulphates increase predicted wine 
quality, while volatile acidity decreases it. Moreover, citric acid contributes minimally 
once other predictors are included. Overall, the model explains a moderate portion of the 
variation, which is indicated by the R² value of 0.322, as shown above. Lastly, the 
residual diagnostics indicate some deviations from normality and mild predictor 
correlation, but the model’s relationships are statistically significant and directionally 
consistent with the expectations.
