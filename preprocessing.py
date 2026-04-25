import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("student_data.csv")

# Separate input and output
X = df.drop(["final_marks", "grade", "risk_level"], axis=1)

y_marks = df["final_marks"]
y_grade = df["grade"]
y_risk = df["risk_level"]

# Encode input categorical columns
le_parent = LabelEncoder()
le_tuition = LabelEncoder()

X["parent_support"] = le_parent.fit_transform(X["parent_support"])
X["tuition"] = le_tuition.fit_transform(X["tuition"])

# Encode output labels
le_grade = LabelEncoder()
le_risk = LabelEncoder()

y_grade_encoded = le_grade.fit_transform(y_grade)
y_risk_encoded = le_risk.fit_transform(y_risk)

# Scale features
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

print("✅ Preprocessing Completed")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# Save scaler and encoders
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_grade, "grade_encoder.pkl")
joblib.dump(le_risk, "risk_encoder.pkl")
joblib.dump(le_parent, "parent_support_encoder.pkl")
joblib.dump(le_tuition, "tuition_encoder.pkl")

print("✅ Scaler and Encoders Saved Successfully")
