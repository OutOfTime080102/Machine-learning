from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# generate list of data points
np.random.seed(22)
means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X1 = np.random.multivariate_normal(means[0], cov, N)
X2 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X1, X2), axis=0).T
y = np.concatenate((np.zeros((1, N)), np.ones((1, N))), axis=1).T
# Xbar
X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)
y_pred = logReg.predict(X)

print(X)
print(y)
print("w: ",logReg.coef_)
print("Độ chính xác accacy: ",accuracy_score(y , y_pred))
print("Độ chính xác confusion_matrix: ",confusion_matrix(y , y_pred))
