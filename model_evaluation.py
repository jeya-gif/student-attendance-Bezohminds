import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("student_data.csv")

# Inputs and outputs
X = df.drop(["final_marks", "grade", "risk_level"], axis=1)

y_marks = df["final_marks"]
y_grade = df["grade"]
y_risk = df["risk_level"]

# Encode categorical inputs
le_parent = LabelEncoder()
le_tuition = LabelEncoder()

X["parent_support"] = le_parent.fit_transform(X["parent_support"])
X["tuition"] = le_tuition.fit_transform(X["tuition"])

# Encode grade and risk outputs
le_grade = LabelEncoder()
le_risk = LabelEncoder()

y_grade_encoded = le_grade.fit_transform(y_grade)
y_risk_encoded = le_risk.fit_transform(y_risk)

# Scale inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_marks_train, y_marks_test = train_test_split(
    X_scaled, y_marks, test_size=0.2, random_state=42
)

X_train2, X_test2, y_grade_train, y_grade_test = train_test_split(
    X_scaled, y_grade_encoded, test_size=0.2, random_state=42
)

X_train3, X_test3, y_risk_train, y_risk_test = train_test_split(
    X_scaled, y_risk_encoded, test_size=0.2, random_state=42
)

# Load trained models
marks_model = tf.keras.models.load_model("marks_model.keras")
grade_model = tf.keras.models.load_model("grade_model.keras")
risk_model = tf.keras.models.load_model("risk_model.keras")

# ------------------- Regression Evaluation -------------------
marks_predictions = marks_model.predict(X_test).flatten()

mae = mean_absolute_error(y_marks_test, marks_predictions)
mse = mean_squared_error(y_marks_test, marks_predictions)
rmse = np.sqrt(mse)

print("\n📌 Regression Evaluation (Final Marks Prediction)")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)

# ------------------- Grade Classification Evaluation -------------------
grade_probs = grade_model.predict(X_test2)
grade_predicted = np.argmax(grade_probs, axis=1)

print("\n📌 Grade Classification Report")
print(classification_report(y_grade_test, grade_predicted, target_names=le_grade.classes_))

print("Confusion Matrix (Grade):")
print(confusion_matrix(y_grade_test, grade_predicted))

# ------------------- Risk Classification Evaluation -------------------
risk_probs = risk_model.predict(X_test3)
risk_predicted = np.argmax(risk_probs, axis=1)

print("\n📌 Risk Classification Report")
print(classification_report(y_risk_test, risk_predicted, target_names=le_risk.classes_))

print("Confusion Matrix (Risk):")
print(confusion_matrix(y_risk_test, risk_predicted))
