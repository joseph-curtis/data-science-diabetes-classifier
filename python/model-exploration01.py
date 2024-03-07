# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:28:36 2024

@author: Joseph Curtis
"""

import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection, svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

filepath = r"C:\Users\joecu\OneDrive - Western Governors University\C964 Computer Science CAPSTONE\Josph_Curtis-Data-Project\diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
# names = ['names-go-here-when-data-is-not-labeled']
df = pd.read_csv(filepath)

mylog_model = linear_model.LogisticRegression(max_iter=10000)
mysvm_model = svm.SVC(max_iter=10000)

y = df.values[:, 0]
X = df.values[:, 1:21]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

mylog_model.fit(X_train, y_train)
mysvm_model.fit(X_train, y_train)

y_pred_log = mylog_model.predict(X_test)
y_pred_svm = mysvm_model.predict(X_test)


print("Logistic Regression prediction results:")
print(metrics.accuracy_score(y_test, y_pred_log))
print("Support Vector Machine model prediction results:")
print(metrics.accuracy_score(y_test, y_pred_svm))

# instructor used this line
# metrics.plot_confusion_matrix(mysvm_model, X_test, y_test)

disp_log = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log)
disp_svm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)
disp_log.plot()
disp_svm.plot()
plt.show()

df.hist()
pyplot.show()
scatter_matrix(df)

"""
# User can input values, model predicts outcome:
values = input("Enter 21? true/false and BMI values seperated by spaces:\n")
valueList = list(values.split(" "))
print(mylog_model.predict([valueList]))
"""
