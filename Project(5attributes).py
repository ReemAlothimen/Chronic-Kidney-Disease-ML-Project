# -*- coding: utf-8 -*-
"""
Created on Sat Dec 2 12:14:27 2023
@author: alothimen
"""

# Importing libraries and tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import plotly.express as px

# Load the dataset
df = pd.read_csv('cleaned_kidney_disease.csv')
X = df.iloc[:, [1, 2, 12, 13, 16]].values
y = df.iloc[:, 25].values

# Extracting categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# =============================================================================
#                        Classification Algorithms
#       KNN
#       Logistic Regression
#       SVM
#       Random Forest
# =============================================================================
# =============================================================================
#                           K-Nearest Neighbors
# =============================================================================

# Create a KNN model
KNNmodel = KNeighborsClassifier(n_neighbors=5, p=2)

# Fit the model on the training data
KNNmodel.fit(X_train, y_train)

# Make predictions for the test set
y_pred = KNNmodel.predict(X_test)

# Calculate the confusion matrix
KNNcm = confusion_matrix(y_test, y_pred)
print(KNNcm)
disp = ConfusionMatrixDisplay(confusion_matrix=KNNcm, display_labels=KNNmodel.classes_)
disp.plot(cmap=plt.cm.pink)
plt.title('KNN')
plt.show()

# Classification report
KNN_accuracy = accuracy_score(y_test, y_pred)
classRep1 = classification_report(y_test, y_pred)
print(classRep1)

# =============================================================================
#                           Logistic Regression
# =============================================================================

# Training the Logistic Regression model on the Training set
LogRegmodel = LogisticRegression(random_state=0)
LogRegmodel.fit(X_train, y_train)

# Predicting the test set results
y_pred = LogRegmodel.predict(X_test)

# Classification report
logreg_acc = accuracy_score(y_test, y_pred)
classRep2 = classification_report(y_test, y_pred)
print(classRep2)

# Confusion Matrix
LogRegcm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=LogRegcm, display_labels=LogRegmodel.classes_)
disp.plot(cmap=plt.cm.pink)
plt.title('Logistic Regression')
plt.show()
print(LogRegcm)


# =============================================================================
#                       Support Vector Machine (SVM)
# =============================================================================

# Create SVM model
SVMmodel = svm.SVC(kernel = 'linear')
SVMmodel.fit(X_train, y_train)

# Make predictions for the test set
y_pred = SVMmodel.predict(X_test)

# Classification report
SVMacc = accuracy_score(y_test, y_pred)
classRep3 = classification_report(y_test, y_pred)
print(classRep3)

# Confusion Matrix
SVMcm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=SVMcm,display_labels=SVMmodel.classes_)
disp.plot(cmap=plt.cm.pink)
plt.title('SVM')
plt.show()
print(SVMcm)

# =============================================================================
#                            Random Forest
# =============================================================================

# Random Forest model
# Fitting the Random Forest model to the training set
RandFormodel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RandFormodel.fit(X_train, y_train)

# Make predictions for the test set 
y_pred = RandFormodel.predict(X_test)

# Classification report
RandForacc = accuracy_score(y_test, y_pred)
classRep4 = classification_report(y_test, y_pred)
print(classRep4)

# Confusion Matrix
RandForcm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=RandForcm,display_labels=RandFormodel.classes_)
disp.plot(cmap=plt.cm.pink)
plt.title('Random Forest')
plt.show()
print(RandForcm)


# Model Comparison
modcomp = pd.DataFrame({
    'Model' : [ 'KNN', 'Logistic Regression','SVM', 'Random Forest'],
    'Score' : [KNN_accuracy, logreg_acc, SVMacc, RandForacc]
})

sorted_models = modcomp.sort_values(by = 'Score', ascending = True)
fig = px.bar(data_frame = sorted_models, x= 'Score', y= 'Model',
       title = 'Models Comparison')

fig.show()


