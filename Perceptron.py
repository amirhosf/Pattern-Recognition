# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:01:18 2019

@author: Amirhossein
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

#class defenition
from time import time
from .._base import _BaseModel
from .._base import _IterativeModel
from .._base import _Classifier


class Perceptron(_BaseModel, _IterativeModel, _Classifier):
    
    def __init__(self, eta=0.1, epochs=50, random_seed=None,
                 print_progress=0):

        _BaseModel.__init__(self)
        _IterativeModel.__init__(self)
        _Classifier.__init__(self)

        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, y, init_params=True):
        self._check_target_array(y, allowed={(0, 1)})
        y_data = np.where(y == 0, -1., 1.)

        if init_params:
            self.b_, self.w_ = self._init_params(
                weights_shape=(X.shape[1], 1),
                bias_shape=(1,),
                random_seed=self.random_seed)
            self.cost_ = []

        self.init_time_ = time()
        rgen = np.random.RandomState(self.random_seed)
        for i in range(self.epochs):
            errors = 0

            for idx in self._yield_minibatches_idx(
                    rgen=rgen,
                    n_batches=y_data.shape[0], data_ary=y_data, shuffle=True):

                update = self.eta * (y_data[idx] -
                                     self._to_classlabels(X[idx]))
                self.w_ += (update * X[idx]).reshape(self.w_.shape)
                self.b_ += update
                errors += int(update != 0.0)

            if self.print_progress:
                self._print_progress(iteration=i + 1,
                                     n_iter=self.epochs,
                                     cost=errors)
            self.cost_.append(errors)
        return self

    def _net_input(self, X):
        """ Net input function """
        return (np.dot(X, self.w_) + self.b_).flatten()

    def _to_classlabels(self, X):
        return np.where(self._net_input(X) < 0.0, -1., 1.)

    def _predict(self, X):
        return np.where(self._net_input(X) < 0.0, 0, 1)
    
#defining the main function
dataset_name = "feature"
dataset_label = "label"
df_train = pd.read_csv(dataset_name + "_train.csv")
df_test = pd.read_csv(dataset_name + "_test.csv")
df_train_label =pd.read_csv(dataset_label + "_train.csv")
df_test_label =pd.read_csv(dataset_label + "_test.csv")
df_train.columns = ['f1','f2']
df_test.columns = ['f1','f2']
X = df_test.iloc[:, [0,1]].values
y = y = df_test_label.iloc[:, 0].values
y = np.where(y == 1, -1, 1)
#-----------------------------------
ppn = Perceptron(epochs=5, 
                 eta=0.05, 
                 random_seed=0,
                 print_progress=3)
ppn.fit(X, y)

plot_decision_regions(X, y, clf=ppn)
plt.title('Perceptron - Rosenblatt Perceptron Rule')
plt.show()

print('Bias & Weights: %s' % ppn.w_)

plt.plot(range(len(ppn.cost_)), ppn.cost_)
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()