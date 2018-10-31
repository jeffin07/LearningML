import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv('data/titanic.csv', sep = '\t')
print(data.columns.values)
y = data["Survived"]
new_x = data[['Sex','Pclass',"Fare","Age"]]
new_x["Age"] = new_x["Age"].fillna(new_x["Age"].median())
sex = {"male":0,"female":1}
new_x["Sex"] = new_x["Sex"].apply(lambda x:sex[x])
print(new_x.head())
x_train,x_test,y_train,y_test=train_test_split(new_x,y)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))