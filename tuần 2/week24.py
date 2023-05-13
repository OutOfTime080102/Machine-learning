import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
data = pd.read_csv('C:\\Users\\Bui Duy\\PycharmProjects\\pythonProject\\tuần 2\\vidu4_lin_reg.txt',sep=" ",header=0)
regr = linear_model.LinearRegression()
y_data = data.iloc[:, -1]
x_data = data.iloc[:, 1:6]
#a
regr.fit(x_data, y_data)
res = list(zip(x_data.columns.tolist(), regr.coef_))
print("a.")
for o in res:
    print("{: >20}: {: >10}".format(*o))
#b
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=False)
regr.fit(x_train, y_train)
res = list(zip(x_data.columns.tolist(), regr.coef_))
print("b.")
for o in res:
    print("{: >20}: {: >10}".format(*o))
y_pred = regr.predict(x_test)
print("Kỳ vọng của sai số là: " , mean_absolute_error(y_pred, y_test))
print("Phương sai của sai số là: " ,mean_squared_error(y_pred, y_test))