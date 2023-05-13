import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn import metrics
import time


data = pd.read_csv("Admission_Predict.csv")
data

X = data.iloc[:, 1:8]
X
y = data.iloc[:, 8]
y

X_train = X[:350]
X_test = X[350:]
y_train = y[:350]
y_test = y[350:]

#a.hồi quy Logistic

logreg = LogisticRegression(max_iter = 10000)
y_train_classified = np.where(y_train >= 0.75, 1, 0)
start_time = time.time()
logreg.fit(X_train, y_train_classified)
end_time = time.time()
print("a.Hồi quy Logistic")
print("Training time: ", end_time - start_time)

print("Intercept: ", logreg.intercept_)
print("Coefficients:\n", logreg.coef_)

y_pred = logreg.predict(X_test)

y_test_classified = np.where(y_test >= 0.75, 1, 0)
print("Accuracy:  ", metrics.accuracy_score(y_test_classified, y_pred))
print("Precision: ", metrics.precision_score(y_test_classified, y_pred))
print("Recall:    ", metrics.recall_score(y_test_classified, y_pred))

#b.hồi quy tuyến tính

linreg = LinearRegression()

linreg.fit(X_train, y_train)

print("b.Hồi quy tuyến tính")
print('Intercept: ', linreg.intercept_)
print('Coefficients: ', linreg.coef_)

y_pred = linreg.predict(X_test)

print('MSE: ', metrics.mean_squared_error(y_test, y_pred))

#c.Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import time

model = GaussianNB()

y_train_classified = np.where(y_train >= 0.75, 1, 0)
start_time = time.time()
model.fit(X_train, y_train_classified)
end_time = time.time()
print("c.Naive Bayes")
print(f'Training time: {end_time - start_time} seconds')

y_pred = model.predict(X_test)

y_test_classified = np.where(y_test >= 0.75, 1, 0)
print("Accuracy:", metrics.accuracy_score(y_test_classified, y_pred))
print("Precision:", metrics.precision_score(y_test_classified, y_pred))
print("Recall:", metrics.recall_score(y_test_classified, y_pred))