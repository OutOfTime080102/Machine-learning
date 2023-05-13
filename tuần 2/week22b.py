import math
import numpy as np
import pandas as pd
with open('fuel.txt') as f:
    lines = f.readlines()

x_data = []
y_data = []
lines.pop(0)

for line in lines:
    splitted = line.replace('\n', '').split(',')
    splitted.pop(0)
    splitted = list(map(float, splitted))
    fuel = 1000 * splitted[1] / splitted[5]
    dlic = 1000 * splitted[0] / splitted[5]
    logMiles = math.log2(splitted[3])
    y_data.append([fuel])
    x_data.append([splitted[-1], dlic, splitted[2], logMiles])

x_data = pd.DataFrame(x_data)
y_data = pd.DataFrame(y_data)
Xbar = np.concatenate((np.ones((x_data.shape[0],1)),x_data),axis = 1)
from sklearn import datasets, linear_model
# Load training data here and assign to Xbar (obs. Data) and y (label)
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False)
# fit_intercept = for calculating the bias

regr.fit(Xbar, y_data)
print(regr.coef_)
y_pred = regr.predict(Xbar)
#print(regr.score(y_data,y_pred))
score = np.dot((y_data - y_pred).T,(y_data - y_pred))
print(score)
