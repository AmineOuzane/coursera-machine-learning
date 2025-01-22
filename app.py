# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('rainfall_data.csv')  # Replace with the actual file path

# Data exploration and cleaning
print(data.info())
print(data.describe())

# Handle missing values
data = data.dropna()  # Simplest approach, drop rows with missing values

# Encode categorical variables
label_encoder = LabelEncoder()
data['RainTomorrow'] = label_encoder.fit_transform(data['RainTomorrow'])
data['RainToday'] = label_encoder.fit_transform(data['RainToday'])

# Feature selection (example features; adjust based on your dataset)
features = ['Humidity3pm', 'Temp3pm', 'Pressure3pm', 'WindGustSpeed', 'RainToday']
target = 'RainTomorrow'

X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate Logistic Regression
log_reg_pred = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))

# Build Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate Random Forest Classifier
rf_pred = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Feature importance for Random Forest
importances = rf_clf.feature_importances_
feature_names = features
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()

# Confusion matrix for both models
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, log_reg_pred))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# True Positive Rate for both models
def calculate_tpr(conf_matrix):
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    tpr = tp / (tp + fn) * 100
    return round(tpr, 2)

log_reg_tpr = calculate_tpr(confusion_matrix(y_test, log_reg_pred))
rf_tpr = calculate_tpr(confusion_matrix(y_test, rf_pred))

print(f"Logistic Regression True Positive Rate: {log_reg_tpr}%")
print(f"Random Forest True Positive Rate: {rf_tpr}%")
