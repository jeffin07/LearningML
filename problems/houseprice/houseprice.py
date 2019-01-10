import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns


data = pd.read_csv('../data/train.csv')
# area, yearbuilt
y = data.iloc[:, -1:].values
'''
x = data.iloc[:, 4:5]
print(type(x))
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
print(model.score(x_test, y_test))
'''

# f1 = np.array(data[['LotArea', 'YearBuilt']])
# print(f1)

# x_train, x_test, y_train, y_test = train_test_split(f1, y)

# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(r2_score(y_test, y_pred))
# print(model.score(x_test, y_test))

print(data.info())
# x = data.drop('SalePrice', 1)
# y = np.log(data.SalePrice)

# x_train, x_test, y_train, y_test = train_test_split(x, y)

# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(r2_score(y_test, y_pred))
# print(model.score(x_test, y_test))

print(data.Foundation.value_counts())
sns.countplot(data.Foundation)
plt.show()
