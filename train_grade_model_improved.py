import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("student_data.csv")

# Split input and output
X = df.drop(["final_marks", "grade", "risk_level"], axis=1)
y_grade = df["grade"]

# Encode input categorical features
le_parent = LabelEncoder()
le_tuition = LabelEncoder()

X["parent_support"] = le_parent.fit_transform(X["parent_support"])
X["tuition"] = le_tuition.fit_transform(X["tuition"])

# Encode output grade labels
le_grade = LabelEncoder()
y_encoded = le_grade.fit_transform(y_grade)

# Compute class weights (to handle imbalance)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)

class_weights_dict = dict(enumerate(class_weights))

print("✅ Class Weights:", class_weights_dict)

# One-hot encoding
y_onehot = to_categorical(y_encoded)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)

# Improved ANN model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(y_onehot.shape[1], activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model using class weights
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=32,
    class_weight=class_weights_dict
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\n✅ Improved Grade Model Evaluation:")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save improved model
model.save("grade_model.keras")

# Save scaler and encoders (for Streamlit)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_grade, "grade_encoder.pkl")
joblib.dump(le_parent, "parent_support_encoder.pkl")
joblib.dump(le_tuition, "tuition_encoder.pkl")

print("\n✅ Improved Grade Model Saved Successfully as grade_model.keras")
