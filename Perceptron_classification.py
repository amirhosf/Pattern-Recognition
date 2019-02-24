# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:42:52 2019

@author: Amirhossein Forouzani
"""
import csv
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from mlxtend.plotting import plot_decision_regions
#defining a class for OOP

class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)        
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    #Defining the main function for file import
dataset_name = "synthetic2"
#dataset_label = "label"
df_train = pd.read_csv(dataset_name + "_train.csv")
df_test = pd.read_csv(dataset_name + "_test.csv")
#df_train_label =pd.read_csv(dataset_label + "_train.csv")
#df_test_label =pd.read_csv(dataset_label + "_test.csv")
df_train.columns = ['f1','f2','labels']
df_test.columns = ['f1','f2','labels']

X = df_test.iloc[:, [0,1]].values

y = df_test.iloc[:, 2].values
#print (y)
y = np.where(y == 1, -1, 1)
y1 = y
#print(y)

ppn = Perceptron(epochs=20, eta=0.1)

ppn.train(X, y)
error_train = 1 - accuracy_score(y, y1)
print (error_train)
#error_test = 1 - accuracy_score(y_test, y_pred_test)
print('Weights: %s' % ppn.w_)
plot_decision_regions(X, y, clf=ppn)
plt.title('Perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()
