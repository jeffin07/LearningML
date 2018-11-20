import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv('../data/train.csv')
y=data.iloc[:,-1:].values
x = data.iloc[:,4:5]
print(type(x))
model = LinearRegression()
x_train, x_test, y_train, y_test =train_test_split(x,y)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test,y_pred))
print(model.score(x_test,y_test))