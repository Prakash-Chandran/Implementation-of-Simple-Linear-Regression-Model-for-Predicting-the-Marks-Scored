# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAKASH C
RegisterNumber:  212223240122
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### Dataset

![Screenshot 2024-08-30 135713](https://github.com/user-attachments/assets/68dcb42e-da96-44de-be64-2fb5306fd842)

### Head Values

![Screenshot 2024-08-30 135727](https://github.com/user-attachments/assets/95562721-c22e-48fe-aa52-54542a08d1f4)

### Tail Values

![Screenshot 2024-08-30 135750](https://github.com/user-attachments/assets/50131f8f-b1a5-4367-9ae5-d2059bcf73cb)

### X and Y values

![Screenshot 2024-08-30 140012](https://github.com/user-attachments/assets/ed12b651-3109-401b-b5c3-49af3eab2568)

### Predication values of X and Y

![Screenshot 2024-08-30 140019](https://github.com/user-attachments/assets/3e06083a-55d8-4b31-90d2-8bf440e50133)

### MSE,MAE and RMSE

![Screenshot 2024-08-30 140108](https://github.com/user-attachments/assets/dc2ab63f-d868-4bfc-a5be-77c0dbf19cba)

### Training Set

![download](https://github.com/user-attachments/assets/3498e675-64a5-402c-99e9-aa0c1721312e)

### Testing Set

![download (1)](https://github.com/user-attachments/assets/5f91ea47-a46f-452c-9111-3a7f35e92b02)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
