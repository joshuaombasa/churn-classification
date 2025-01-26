# model.py - Churn Classification Model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
churn_df = pd.read_csv("churn_df.csv")  # Make sure to place your dataset in the same directory

# Drop unnecessary columns
churn_df.drop(columns=["CustomerID"], errors="ignore", inplace=True)

# Convert categorical variables to numerical
label_encoders = {}
for col in churn_df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    churn_df[col] = le.fit_transform(churn_df[col])
    label_encoders[col] = le

# Split data into features (X) and target (y)
X = churn_df.drop(columns=["Churn"])  # Features
y = churn_df["Churn"]  # Target (1 = Churn, 0 = Not Churn)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
