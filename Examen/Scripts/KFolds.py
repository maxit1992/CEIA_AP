# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:27:30 2021

@author: MaxiT
"""

import numpy as np

def kfolds(X, y, k):
    
    folds = []
    chunk_size = round(X.shape[0] / k)
    permidx=np.random.permutation(X.shape[0])
    for i in range(0, X.shape[0], chunk_size):
        end = i + chunk_size if i + chunk_size <= X.shape[0] else X.shape[0]
        x_valid = X[permidx[i: end]]
        y_valid = y[permidx[i: end]]
        x_train = np.concatenate([X[permidx[: i]], X[permidx[end:]]])
        y_train = np.concatenate([y[permidx[: i]], y[permidx[end:]]])


        folds.append((x_train,y_train,x_valid,y_valid))

    return folds



#x = np.random.randint(0,255,size=(1000,15)) 
#y = np.random.randint(0,2,size=(1000))
#for x_train,y_train,x_valid,y_valid in kfolds(x,y,10):
#    print("-------------")
#    print("x train",x_train.shape)
#    print("y train",y_train.shape)
#    print("x valid",x_valid.shape)
#    print("y valid",y_valid.shape)