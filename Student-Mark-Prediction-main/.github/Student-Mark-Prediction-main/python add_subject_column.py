import pandas as pd

# Use the full path to the CSV
df = pd.read_csv("D:/Student-Mark-Prediction-main/Student-Mark-Prediction-main/student-mat.csv")

subjects = ["OS", "DIP", "SE", "FAI", "FLAT"]
df["subject"] = [subjects[i % len(subjects)] for i in range(len(df))]

df.to_csv("D:/Student-Mark-Prediction-main/Student-Mark-Prediction-main/student-mat-with-subject.csv", index=False)

print("New dataset saved as student-mat-with-subject.csv with subject column added.")
