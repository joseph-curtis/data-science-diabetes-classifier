# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 16:59:39 2024

@author: Joseph Curtis
"""

import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import KFold, cross_val_score

# Load the CSV file
filepath_2015 = r"C:\Users\joecu\OneDrive - Western Governors University\C964 Computer Science CAPSTONE\Josph_Curtis-Data-Project\diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df1 = pd.read_csv(filepath_2015)

# # Display the first few rows of the dataframe to understand its structure
# df1.head()

# Load the second CSV file
filepath_2021 = r"C:\Users\joecu\OneDrive - Western Governors University\C964 Computer Science CAPSTONE\Josph_Curtis-Data-Project\diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
df2 = pd.read_csv(filepath_2021)

# Combine the two DataFrames
combined_df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

# # Display the first few rows of the combined dataframe and its shape to verify 
# # the combination
# combined_df_info = combined_df.head(), combined_df.shape

# display combined_df_info


# Remove irrelevant features from the combined dataset
columns_to_remove = ['CholCheck', 'AnyHealthcare', 'NoDocbcCost', 'Education', 'Income']
reduced_df = combined_df.drop(columns=columns_to_remove)

# # Display the first few rows of the reduced dataframe to verify the removal
# reduced_df.head()

# # Check for missing values in the reduced dataset
# missing_values = reduced_df.isnull().sum()

# missing_values ## no missing values found

# # Check the range of values for the specified features to determine suitable data types
# features_to_optimize = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']
# data_types_optimization = reduced_df[features_to_optimize].describe().loc[['min', 'max']]

# data_types_optimization

# # Before reduced data types
# memory1 = reduced_df.memory_usage(index=True).sum()

binary_columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'DiffWalk', 'Sex']
for column in binary_columns:
    reduced_df[column] = reduced_df[column].astype('bool')

# # After data types reduces memory size
# memory2 = reduced_df.memory_usage(index=True).sum()

###### Logistic Regression algorithm ######

# copy Dataframe for Logistic model
log_df = reduced_df.copy(deep=True)

# Selecting numerical columns (excluding binary/boolean columns)
numerical_columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical features
log_df[numerical_columns] = scaler.fit_transform(log_df[numerical_columns])

mylog_model = linear_model.LogisticRegression(solver='saga', max_iter=1000)

# 'X' is the feature set and 'y' is the target variable
X_log = log_df.drop('Diabetes_binary', axis=1)
y_log = log_df['Diabetes_binary'].astype('bool')  # Ensuring the target is boolean

# Splitting the dataset into the Training set and Test set
X_log_train, X_log_test, y_log_train, y_log_test = model_selection.train_test_split(X_log, y_log, test_size=0.25, random_state=42)

mylog_model.fit(X_log_train, y_log_train)

y_pred_log = mylog_model.predict(X_log_test)

# Evaluate the model
k_folds = KFold(n_splits = 5, shuffle=True)
# The number of folds determines the test/train split for each iteration. 
# So 5 folds has 5 different mutually exclusive training sets. 
# That's a 1 to 4 (or .20 to .80) testing/training split for each of the 5 iterations.

log_scores = cross_val_score(mylog_model, X_log, y_log)
# This shows the average score. Print 'scores' to see an array of individual iteration scores.
print("Logistic Regression Average Prediction Score: ", log_scores.mean())

accuracy_log = accuracy_score(y_log_test, y_pred_log)
conf_matrix_log = confusion_matrix(y_log_test, y_pred_log)
class_report_log = classification_report(y_log_test, y_pred_log)

print("\nLogistic Regression (single) prediction results:")
print(f"Accuracy: {round(accuracy_log*100,2)} %")
print("Confusion Matrix:")
print(conf_matrix_log)
print("Classification Report:")
print(class_report_log)

######  Random Forest algorithm ######

# rename existing Dataframe for RF model
rforest_df = reduced_df.copy(deep=False)

# scale data types down to reduce memory footprint
rforest_df['BMI'] = rforest_df['BMI'].astype('float32')
# Use 'float32' if 'BMI' can have decimal values
rforest_df['GenHlth'] = rforest_df['GenHlth'].astype('int8')
rforest_df['MentHlth'] = rforest_df['MentHlth'].astype('int8')
rforest_df['PhysHlth'] = rforest_df['PhysHlth'].astype('int8')
rforest_df['Age'] = rforest_df['Age'].astype('int8')

# 'X' is the set of features and 'y' is the target variable
X_rf = rforest_df.drop('Diabetes_binary', axis=1)
y_rf = rforest_df['Diabetes_binary'].astype('bool')  # Ensuring the target is boolean

# Splitting the dataset into the Training set and Test set
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.25, random_state=42)

# Creating a Random Forest Classifier -- You can adjust parameters
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting Random Forest to the Training set
rf_classifier.fit(X_rf_train, y_rf_train)

# Predicting the Test set results
y_pred_rf = rf_classifier.predict(X_rf_test)

# Evaluating the results
k_folds = KFold(n_splits = 5, shuffle=True)
# The number of folds determines the test/train split for each iteration. 
# So 5 folds has 5 different mutually exclusive training sets. 
# That's a 1 to 4 (or .20 to .80) testing/training split for each of the 5 iterations.

rf_scores = cross_val_score(rf_classifier, X_rf, y_rf)
# This shows the average score. Print 'scores' to see an array of individual iteration scores.
print("Random Forest Average Prediction Score: ", rf_scores.mean())

accuracy_rf = accuracy_score(y_rf_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_rf_test, y_pred_rf)
class_report_rf = classification_report(y_rf_test, y_pred_rf)

print("\nRandom Forest (single) prediction results:")
print(f"Accuracy: {round(accuracy_rf*100,2)} %")
print("Confusion Matrix:")
print(conf_matrix_rf)
print("Classification Report:")
print(class_report_rf)






# Plot graphs
disp_log = ConfusionMatrixDisplay.from_predictions(y_log_test, y_pred_log)
disp_rforest = ConfusionMatrixDisplay.from_predictions(y_rf_test, y_pred_rf)
disp_log.plot()
disp_rforest.plot()
# plt.show()

rforest_df.hist()
pyplot.show()
