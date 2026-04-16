# Disease Prediction Model — Diabetes
# By Padma Shree
# Project 8 of 25 — FIRST MACHINE LEARNING PROJECT!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("=" * 50)
print("🩺 DISEASE PREDICTION MODEL — DIABETES")
print("=" * 50)

# Step 1: Load data
file_path = r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\diabetes.csv"

try:
    df = pd.read_csv(file_path)
    print(f"\n✅ Dataset loaded: {len(df)} patients")
    print(f"Features: {df.columns.tolist()}")
except FileNotFoundError:
    print("\n❌ File not found! Please download diabetes.csv from Kaggle")
    print("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    exit()

# Step 2: Understand the data
print(f"\n📊 Data shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nFirst 5 patients:\n{df.head()}")

# Step 3: Split features (X) and target (y)
# Outcome = 1 (has diabetes), 0 (no diabetes)
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target

print(f"\n🎯 Patients with diabetes: {y.sum()}")
print(f"✅ Patients without diabetes: {len(y) - y.sum()}")

# Step 4: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📚 Training data: {len(X_train)} patients")
print(f"🧪 Testing data: {len(X_test)} patients")

# Step 5: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n🤖 Model trained successfully!")

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

# Step 9: Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

print(f"\n🔍 Most important factors for diabetes:")
print(feature_importance)

print("\n" + "=" * 50)
print("✅ PROJECT 8 COMPLETE! First ML model working!")
print("=" * 50)