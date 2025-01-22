# Final Project: Rainfall Prediction Classifier

## 1. Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


## 2. Load the Dataset
# Replace 'rainfall_data.csv' with the actual dataset file path
data = pd.read_csv('rainfall_data.csv')

# Initial Exploration
print(data.info())
print(data.describe())


## 3. Handle Missing Values
# Dropping rows with missing values (simple approach)
data = data.dropna()


## 4. Encode Categorical Variables
# Encoding categorical features
label_encoder = LabelEncoder()
data['RainTomorrow'] = label_encoder.fit_transform(data['RainTomorrow'])
data['RainToday'] = label_encoder.fit_transform(data['RainToday'])

## 5. Feature Selection
# Selecting features for the model
features = ['Humidity3pm', 'Temp3pm', 'Pressure3pm', 'WindGustSpeed', 'RainToday']
target = 'RainTomorrow'

X = data[features]
y = data[target]

## 6. Split the Data into Training and Testing Sets
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## 7. Standardize Features
# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## 8. Train Logistic Regression Model

# Training Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


## 9. Evaluate Logistic Regression Model
# Predictions and evaluation for Logistic Regression
log_reg_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))

## 10. Train Random Forest Classifier

# Training Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)


## 11. Evaluate Random Forest Classifier
# Predictions and evaluation for Random Forest
rf_pred = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))


## 12. Feature Importance Visualization
# Visualizing feature importance for Random Forest
importances = rf_clf.feature_importances_
feature_names = features
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()

## 13. Confusion Matrices
# Confusion matrices for both models
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, log_reg_pred))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

## 14. Calculate True Positive Rate (TPR)
# Function to calculate TPR
def calculate_tpr(conf_matrix):
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    tpr = tp / (tp + fn) * 100
    return round(tpr, 2)

log_reg_tpr = calculate_tpr(confusion_matrix(y_test, log_reg_pred))
rf_tpr = calculate_tpr(confusion_matrix(y_test, rf_pred))

print(f"Logistic Regression True Positive Rate: {log_reg_tpr}%")
print(f"Random Forest True Positive Rate: {rf_tpr}%")


## 15. Summary and Recommendations
# Summary of findings
print("Summary:")
print(f"Accuracy of Logistic Regression: {accuracy_score(y_test, log_reg_pred)}")
print(f"Accuracy of Random Forest: {accuracy_score(y_test, rf_pred)}")
print(f"True Positive Rate (Logistic Regression): {log_reg_tpr}%")
print(f"True Positive Rate (Random Forest): {rf_tpr}%")

print("Recommendations:")
if rf_tpr > log_reg_tpr:
    print("The Random Forest Classifier is a better model for predicting rainfall tomorrow due to its higher True Positive Rate.")
else:
    print("The Logistic Regression model might be preferable for simplicity, but its TPR is lower than Random Forest.")