# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the required packages and print the present data.
 
 2.Print the placement data and salary data.
 
 3.Find the null and duplicate values.
 
 4.Using logistic regression find the predicted values of accuracy , confusion matrices.
 
 5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kowshika.R 
RegisterNumber: 212224220049
*/
```
```
 import pandas as pd
 from sklearn.preprocessing import LabelEncoder
 from sklearn.model_selection import train_test_split
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import accuracy_score, confusion_matrix, 
classification_report
 # Load dataset
 data = pd.read_csv("Placement_Data.csv")
 # Show first 5 rows of raw data
 print("data.head")
 print(data.head(), "\n")
 # Copy dataset and drop sl_no & salary
 data1 = data.copy()
 data1 = data1.drop(["sl_no", "salary"], axis=1)
 print("data1.head")
 print(data1.head(), "\n")
 # Check for missing values
 print("isnull()")
 print(data1.isnull().sum(), "\n")
 # Check for duplicate rows
 print("data duplicate ")
 print(data1.duplicated().sum(), "\n")
 # Label Encoding categorical features
 le = LabelEncoder()
 for col in ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", 
"specialisation", "status"]:
 data1[col] = le.fit_transform(data1[col])
 print("data")
 print(data1.head(), "\n")
 # Features and Target
 X = data1.iloc[:, :-1]
 y = data1["status"]
print("status")
 print(y.head(), "\n")
 # Train/Test Split
 X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=0
 )
 # Logistic Regression
 lr = LogisticRegression(solver="liblinear")
 lr.fit(X_train, y_train)
 # Predictions
 y_pred = lr.predict(X_test)
 print("y_pred")
 print(y_pred, "\n")
 # Accuracy
 print(" Accuracy Score")
 print(accuracy_score(y_test, y_pred), "\n")
 # Confusion Matrix
 print("Confusion Matrix")
 print(confusion_matrix(y_test, y_pred), "\n")
 # Classification Report
 print("Classification")
 print(classification_report(y_test, y_pred), "\n")
 # Custom Prediction Example
 sample = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
 print("LR predict")
 print(lr.predict(sample))
```

## Output:

## Data.head
<img width="749" height="318" alt="image" src="https://github.com/user-attachments/assets/3e599c3e-d190-4328-b84e-a667965164c8" />

## Data1.head
<img width="748" height="302" alt="image" src="https://github.com/user-attachments/assets/db0ede12-a368-487c-89c1-1691ccd6957f" />

## isnull
<img width="430" height="421" alt="image" src="https://github.com/user-attachments/assets/935a999b-a5f8-480f-807b-ef1e0ea06a4a" />

## Data duplicate
<img width="233" height="88" alt="image" src="https://github.com/user-attachments/assets/a20677a3-97ac-48cc-b4ea-ae36587a8830" />

## Data
<img width="739" height="318" alt="image" src="https://github.com/user-attachments/assets/ecf2f02c-398e-4cbd-8f66-8e526f8bd4d4" />

## Status
<img width="452" height="205" alt="image" src="https://github.com/user-attachments/assets/4a484a96-a418-4fa1-82ca-305b3c69201f" />

## y_pred
<img width="731" height="68" alt="image" src="https://github.com/user-attachments/assets/e66828f3-1d73-4c83-8e75-6c669067cf67" />

## Accuracy
<img width="261" height="68" alt="image" src="https://github.com/user-attachments/assets/c331e449-4a19-4538-971e-aef2833f104c" />

##  Confusion Matrix
<img width="250" height="90" alt="image" src="https://github.com/user-attachments/assets/2582a4c3-cacd-42c5-85f2-73a867875185" />

##  Classification
<img width="734" height="277" alt="image" src="https://github.com/user-attachments/assets/93621f41-3603-4044-955a-a623c30b7819" />

## LR predict
<img width="186" height="61" alt="image" src="https://github.com/user-attachments/assets/8c3b18ae-186d-4953-94d6-3752d2289da6" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
