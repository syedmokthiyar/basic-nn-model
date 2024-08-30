# EX:01 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along. A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions. This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs. Hidden layer -2 input layer -1 outpur layer -1

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

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd  

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Deeplearning').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input': 'int', 'Output': 'int'})
dataset1.head()

x = dataset1[['Input']].values
y = dataset1[['Output']].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=33)

Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train1 = Scaler.transform(x_train)

ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs = 2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
x_n1=[[4]]
x_n1_1 = Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)

```

## Dataset Information
![Screenshot 2024-08-23 082326](https://github.com/user-attachments/assets/e171a9c9-77a9-4573-ae64-fb7f26bb349d)

### OUTPUT

## Training Loss Vs Iteration Plot
![Screenshot 2024-08-23 085725](https://github.com/user-attachments/assets/030070cb-9f5f-4073-9972-75c4fd29610b)

## Test Data Root Mean Squared Error
![Screenshot 2024-08-23 085822](https://github.com/user-attachments/assets/f15b7a7a-bd06-4d96-a935-8e4903d91535)

## New Sample Data Prediction
![Screenshot 2024-08-23 085853](https://github.com/user-attachments/assets/4fa3c7ec-1024-4a0c-9141-71a650b8f147)

## RESULT
Thus a Neural Network regression model for the given dataset is written and executed successfully
