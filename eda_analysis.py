import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_data.csv")

# Histogram
plt.hist(df["final_marks"], bins=20)
plt.title("Final Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.show()

# Attendance vs Marks
plt.scatter(df["attendance"], df["final_marks"])
plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.show()

# Study hours vs Marks
plt.scatter(df["study_hours"], df["final_marks"])
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Marks")
plt.show()

# Stress vs Marks
plt.scatter(df["stress_level"], df["final_marks"])
plt.title("Stress Level vs Final Marks")
plt.xlabel("Stress Level (1-10)")
plt.ylabel("Final Marks")
plt.show()

# Internet vs Marks
plt.scatter(df["internet_hours"], df["final_marks"])
plt.title("Internet Hours vs Final Marks")
plt.xlabel("Internet Hours per Day")
plt.ylabel("Final Marks")
plt.show()

# Grade distribution
grade_counts = df["grade"].value_counts()
plt.bar(grade_counts.index, grade_counts.values)
plt.title("Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.show()

# Risk distribution
risk_counts = df["risk_level"].value_counts()
plt.bar(risk_counts.index, risk_counts.values)
plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Students")
plt.show()

# Correlation matrix
correlation = df.select_dtypes(include=["int64", "float64"]).corr()

plt.figure(figsize=(10, 6))
plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Matrix")
plt.show()
