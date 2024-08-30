# EX:01 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error.

## Neural Network Model
![55](https://github.com/user-attachments/assets/07719be9-b699-45f3-a3a9-7e281538c015)

## DESIGN STEPS

## STEP 1:
Loading the dataset

## STEP 2:
Split the dataset into training and testing

## STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

## STEP 4:
Build the Neural Network Model and compile the model.

## STEP 5:
Train the model with the training data.

## STEP 6:
Plot the performance plot

## STEP 7:
Evaluate the model with the testing data.

## PROGRAM
```
Name: Syed Mokthiyar S M
Register Number: 212222230156
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

!pip install --upgrade gspread

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds,_=default()
gc = gspread.authorize(creds)

worksheet = gc.open('deep learning').sheet1

data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'int'})
dataset1 = dataset1.astype({'Output':'int'})

dataset1.head() 

dataset1.head()
X = dataset1[['input']].values
y = dataset1[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)

MinMaxScaler()
X_train1 = Scaler.transform(X_train)
ai_brain=Sequential([Dense(units=3,input_shape=[1]),Dense(units=3),Dense(units=1)])
ai_brain.compile(optimizer="rmsprop",loss="mae")
ai_brain.fit(X_train1,y_train,epochs=1000)
loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[18]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

ai_brain.evaluate(X_test1,y_test)

ai_brain.predict(X_n1_1)
```

## Dataset Information
![Screenshot 2024-08-23 082326](https://github.com/user-attachments/assets/e171a9c9-77a9-4573-ae64-fb7f26bb349d)

### OUTPUT

## Training Loss Vs Iteration Plot
![Screenshot 2024-08-30 082104](https://github.com/user-attachments/assets/2c9e529b-957e-4012-977a-468f61888578)

## Test Data Root Mean Squared Error
![Screenshot 2024-08-30 082231](https://github.com/user-attachments/assets/b2cd686c-e02f-43a1-9d80-b5c42ab6d272)

## New Sample Data Prediction
![Screenshot 2024-08-30 082333](https://github.com/user-attachments/assets/b64e9bcc-8c04-476e-a5db-4edf27ab7159)

## RESULT
Thus a Neural Network regression model for the given dataset is written and executed successfully
