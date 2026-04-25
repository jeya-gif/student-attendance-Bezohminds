import pandas as pd
import numpy as np

np.random.seed(42)

records = 3000

# Generate input features
attendance = np.random.randint(40, 101, records)  # 40 to 100%
study_hours = np.round(np.random.uniform(0, 10, records), 1)
previous_marks = np.random.randint(35, 101, records)
assignment_score = np.random.randint(30, 101, records)
internal_marks = np.random.randint(10, 51, records)  # out of 50

sleep_hours = np.round(np.random.uniform(3, 10, records), 1)
internet_hours = np.round(np.random.uniform(0, 10, records), 1)
stress_level = np.random.randint(1, 11, records)  # 1 to 10

parent_support = np.random.choice(["Yes", "No"], records, p=[0.7, 0.3])
tuition = np.random.choice(["Yes", "No"], records, p=[0.6, 0.4])

## Convert Yes/No into numeric for calculation
parent_support_num = np.where(parent_support == "Yes", 3, -2)
tuition_num = np.where(tuition == "Yes", 3, -1)

# Updated realistic formula (less bias to 100)
final_marks = (
    (attendance * 0.15) +
    (study_hours * 4) +
    (previous_marks * 0.30) +
    (assignment_score * 0.20) +
    (internal_marks * 0.60) +
    (sleep_hours * 1.5) -
    (internet_hours * 2.5) -
    (stress_level * 2.5) +
    parent_support_num +
    tuition_num +
    np.random.normal(0, 8, records)  # more randomness
)

# Clip between 0 and 100
final_marks = np.clip(final_marks, 0, 100)
final_marks = np.round(final_marks, 2)


# Grade generation
def get_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 75:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 40:
        return "D"
    else:
        return "F"

grades = [get_grade(m) for m in final_marks]

# Risk level generation
def get_risk(marks):
    if marks < 40:
        return "High"
    elif marks < 60:
        return "Medium"
    else:
        return "Low"

risk_levels = [get_risk(m) for m in final_marks]

# Create DataFrame
df = pd.DataFrame({
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_marks": previous_marks,
    "assignment_score": assignment_score,
    "internal_marks": internal_marks,
    "sleep_hours": sleep_hours,
    "internet_hours": internet_hours,
    "stress_level": stress_level,
    "parent_support": parent_support,
    "tuition": tuition,
    "final_marks": final_marks,
    "grade": grades,
    "risk_level": risk_levels
})

# Save dataset
df.to_csv("student_data.csv", index=False)

print("✅ Dataset created successfully: student_data.csv")
print(df.head())
 