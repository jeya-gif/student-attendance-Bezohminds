import pandas as pd

df = pd.read_csv("student_data.csv")

print("✅ Dataset Loaded Successfully")
print(df.head())

print("\nDataset Shape (rows, columns):")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nDuplicate Rows Count:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print(df.shape)

print("\nStatistical Summary:")
print(df.describe())

print("\nGrade Distribution:")
print(df["grade"].value_counts())

print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())
