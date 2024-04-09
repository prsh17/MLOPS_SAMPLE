import numpy as np
import pandas as pd  # Library for data manipulation
from sklearn.model_selection import train_test_split  # Function for splitting data into training and test sets
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import confusion_matrix, classification_report  # Functions for evaluating classifier performance

# Load the dataset using pandas library
df = pd.read_csv('BreastCancer.csv')

# (a) Find which attributes should be trivially excluded for classification
# Remove the 'Id' column as it is an identifier
df_cleaned = df.drop(columns=['Id'])

# (b) Data Cleaning (handle missing values, encode categorical data if any, etc.)
# Check for missing values in the dataset
missing_values = df_cleaned.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values by filling them with the median of each column
numeric_columns = df_cleaned.select_dtypes(include=['number']).columns
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())

# Check if there are any remaining missing values after cleaning
missing_values_after_cleaning = df_cleaned.isnull().sum()
print("Missing Values After Cleaning:\n", missing_values_after_cleaning)

# (c) Divide the data into Training Set (70%) and Test Set (30%) using Stratified Sampling
# Split the dataset into features (X) and target variable (y)
X = df_cleaned.drop(columns=['Class'])  # Features
y = df_cleaned['Class']  # Target variable

# Split the data into training and test sets using stratified sampling
# train_test_split function from sklearn library is used for this purpose
# Inputs:
# X: Features
# y: Target variable
# test_size: Size of the test set (in this case, 30%)
# stratify: Ensures that the class distribution is similar in both training and test sets
# random_state: Seed for random number generation to ensure reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=41)

# (2) Using Support Vector Machine (SVM) to train the model using the Training Set.
# Train the SVM classifier
# SVC function from sklearn library is used to create an SVM classifier
# Inputs:
# kernel: Specifies the kernel type (in this case, 'linear' kernel is used)
# gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels (auto is used for 'linear' kernel)
# C: Regularization parameter
classifier_svm = SVC(kernel='linear', gamma='auto', C=1)
classifier_svm.fit(X_train, y_train)

# (3) Write the Model and use the model for testing it using Test Set. Determine the accuracy of the Classifier.
# Predict using the trained SVM model
y_predict = classifier_svm.predict(X_test)

# Confusion matrix
# confusion_matrix function from sklearn.metrics library is used to calculate the confusion matrix
# Inputs:
# y_test: True labels from the test set
# y_predict: Predicted labels by the classifier
c_svm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(c_svm)

# Calculate accuracy
# Accuracy is calculated as the ratio of correct predictions to the total number of predictions
accuracy_svm = sum(np.diag(c_svm)) / np.sum(c_svm)
print("Accuracy:", accuracy_svm)

# Classification report
# classification_report function from sklearn.metrics library is used to generate a classification report
# Inputs:
# y_test: True labels from the test set
# y_predict: Predicted labels by the classifier
print("Classification Report:")
print(classification_report(y_test, y_predict))
