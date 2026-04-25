import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
df = pd.read_csv("student_data.csv")

# Split input and output
X = df.drop(["final_marks", "grade", "risk_level"], axis=1)
y = df["final_marks"]

# Encode categorical inputs
le_parent = LabelEncoder()
le_tuition = LabelEncoder()

X["parent_support"] = le_parent.fit_transform(X["parent_support"])
X["tuition"] = le_tuition.fit_transform(X["tuition"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build ANN model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)

print("\n✅ Regression Model Evaluation:")
print("Test Loss (MSE):", loss)
print("Test MAE:", mae)

# Save model and preprocessing objects
model.save("marks_model.keras")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_parent, "parent_support_encoder.pkl")
joblib.dump(le_tuition, "tuition_encoder.pkl")

print("\n✅ Marks Model Saved Successfully as marks_model.keras")
