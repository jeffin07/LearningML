import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv('../data/train.csv')
y=data.iloc[:,-1:].values
# print(data['Bedroom'])
my_list = ['LotArea', 'Neighborhood', 'BedroomAbvGr', 'PoolArea']
selected = data[my_list]
mean_list = ['LotArea','BedroomAbvGr', 'PoolArea']
x=[]
for i in mean_list:
	selected[i+"N"] = [m/selected[i].mean() for m in selected[i]]
mean_list = ['LotAreaN','Neighborhood', 'BedroomAbvGrN', 'PoolAreaN']
print(selected[mean_list])
Neighborhood_values =  list(np.unique((selected['Neighborhood'])))
print(len(Neighborhood_values))
selected['NeighborhoodN'] = [Neighborhood_values.index(i) for i in selected['Neighborhood']]
mean_list = ['LotAreaN','NeighborhoodN', 'BedroomAbvGrN', 'PoolAreaN']
x=selected[mean_list]

x_train, x_test, y_train, y_test =train_test_split(x,y)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test,y_pred))
for i in range(10):
	print(y_pred[i],y_test[i])