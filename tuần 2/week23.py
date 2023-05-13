import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats

# Change to data path on your computer
data = pd.read_csv("C:\\Users\\Bui Duy\\PycharmProjects\\pythonProject\\SAT_GPA.csv")
# Show the description of data
data.describe()
# Set to training data (x, y)
y = np.array(data['GPA']).reshape(-1, 1)
x = np.array(data['SAT']).reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=60, shuffle=False)
regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
yhat = regr.intercept_ + x * regr.coef_

plt.scatter(x, y)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='y')
fig = plt.plot(x, yhat, lw=4, c='blue', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()
