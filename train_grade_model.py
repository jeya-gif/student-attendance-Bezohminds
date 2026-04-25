import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

# Encode grade output labels
le_grade = LabelEncoder()
y_encoded = le_grade.fit_transform(y_grade)

# Convert labels into one-hot encoding
y_onehot = to_categorical(y_encoded)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)

# Build ANN model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(y_onehot.shape[1], activation="softmax")  # output neurons = number of classes
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\n✅ Grade Model Evaluation:")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save model and encoders
model.save("grade_model.keras")

joblib.dump(scaler, "scaler_grade.pkl")
joblib.dump(le_grade, "grade_encoder.pkl")
joblib.dump(le_parent, "parent_support_encoder.pkl")
joblib.dump(le_tuition, "tuition_encoder.pkl")

print("\n✅ Grade Model Saved Successfully as grade_model.keras")
