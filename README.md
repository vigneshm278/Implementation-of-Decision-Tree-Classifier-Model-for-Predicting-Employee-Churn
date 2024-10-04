# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program

Step 2: attach the given data file

Step 3: now find the satisfaction level of employee data

Step 4: Find the accuracy and new predict value

Step 5: end the program
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: YUVARAJ V
RegisterNumber:  212223230252
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Desktop/INTR MACH/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

data["salary"]= le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level" , "last_evaluation" , "number_project" ,"average_montly_hours" , "time_spend_company" ,"Work_accident" , "promotion_last_5years" , "salary"]]
x.head()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state= 100)

from sklearn.tree  import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion= "entropy")
dt.fit(x_train, y_train)
y_pred= dt.predict(x_test)

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test , y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

![image](https://github.com/user-attachments/assets/a48dd485-4cd1-4b7a-bf84-235dc7bca53c)


![image](https://github.com/user-attachments/assets/51ca152d-93db-4630-8bb0-1ce89b7935aa)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
