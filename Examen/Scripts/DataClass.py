# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:12:32 2021

@author: MaxiT
"""

import numpy as np

class Data(object):
    
    
    instance=None
    dataset=None
    
    #Singleton
    def __new__(cls,path):
        if Data.instance is None:
            Data.instance=super(Data,cls).__new__(cls)
            return Data.instance
        else:
            return Data.instance

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('x_1', np.float),
                     ('x_2', np.float),
                     ('y', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0) #OJO!
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split_train_valid_test(self, pct_valid, pct_test):
        X = np.array([self.dataset['x_1'], self.dataset['x_2']]).T
        y = self.dataset['y']

        permuted_idxs = np.random.permutation(X.shape[0])

        test_idxs = permuted_idxs[0:round(pct_test * X.shape[0])]
        
        valid_idxs = permuted_idxs[round(pct_test * X.shape[0]): \
                                   round((pct_test+pct_valid) * X.shape[0])]

        train_idxs = permuted_idxs[round((pct_test+pct_valid) * X.shape[0]): \
                                   X.shape[0]]

        X_train = X[train_idxs]
        X_valid = X[valid_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_valid = y[valid_idxs]
        y_test = y[test_idxs]

        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def split_train_test(self, pct_test):
        X = np.array([self.dataset['x_1'], self.dataset['x_2']]).T
        y = self.dataset['y']

        permuted_idxs = np.random.permutation(X.shape[0])

        test_idxs = permuted_idxs[0:int(pct_test * X.shape[0])]
        

        train_idxs = permuted_idxs[int(pct_test * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test
    
    

#dataset = Data("./dataset.txt")
#X_train, X_valid, X_test, y_train, y_valid, y_test =\
#    dataset.split_train_valid_test(0.15,0.15)