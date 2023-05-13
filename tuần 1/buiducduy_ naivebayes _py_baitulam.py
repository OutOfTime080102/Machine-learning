from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Loại bỏ các line data thừa
from sklearn.naive_bayes import GaussianNB

path = 'NB_last/'
with open('datacum.txt', 'r') as oFile, open('rawData.csv', 'w') as nFile:
    for line in oFile:
        if not line.startswith('#') and not line.startswith('\n'):
            nFile.write(line)
data = pd.read_csv('rawData.csv', header=None)

B_benign = data[data[1] == 2]  # u lành tính(B-benign)
M_malignant = data[data[1] == 4]  # u ác tính(M-malignant)

testData = pd.concat([B_benign.sample(n=80, random_state=42), M_malignant.sample(n=40, random_state=42)])
trainData = data.drop(testData.index)
x_train = trainData.drop([1], axis=1)
y_train = trainData[1]
gnb = GaussianNB()
gnb.fit(x_train, y_train)
x_test = testData.drop([1], axis=1)
y_test = testData[1]

y_pred = gnb.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=2)
recall = recall_score(y_test, y_pred, pos_label=4)

print("Độ chính xác của mô hình là:")
print("Acurracy:  {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))