# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:28:43 2021

@author: MaxiT
"""

import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork():
    """
    Modelo de red neuronal simple con numpy
    """
    W1 = None
    W2 = None
    W3 = None
    
    def __init__(self):
        pass
    
    
    def sigmoid(self, x):
        g_x = 1 / (1 + np.exp(-x))
        return g_x
        
        
    def fit(self, X, y, lr=0.1, epochs=100000):
        
        X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
        
        n = X.shape[0]
        m = X.shape[1]
        
        # initialize random weights
        W1 = np.random.random(size=(m, 1))
        W2 = np.random.random(size=(m,1))
        W3= np.random.random(size=(3,1))
        
        # iterate over the n_epochs
        for j in range(epochs):

            # Shuffle all the samples 
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

            # Iterate over the batches
            for i in range(n):

                # Forward propagation
                z1 = X[i,:]@W1
                z1 = z1[0]
                a1 = self.sigmoid(z1)
                
                z2 = X[i,:]@W2
                z2 = z2[0]
                a2 = self.sigmoid(z2)
                
                x1_2 = np.array([a1,a2,1])
                z3 = x1_2@W3
                z3 = z3[0]
                a3 = self.sigmoid(z3)
                
                prediction = a3
                
                # Calculate the error 
                error = y[i] - prediction
                
                # Calculate the gradient
                grad_W3 = (-2/n) * error * self.sigmoid(z3) * (1-self.sigmoid(z3)) * x1_2
                grad_W2 = (-2/n) * error * self.sigmoid(z3) * (1-self.sigmoid(z3)) * W3[1] * \
                            self.sigmoid(z2) * (1-self.sigmoid(z2)) * X[i,:]
                grad_W1 = (-2/n) * error * self.sigmoid(z3) * (1-self.sigmoid(z3)) * W3[0] * \
                            self.sigmoid(z1) * (1-self.sigmoid(z1)) * X[i,:]
                
                # Back propagation
                W3 = W3 - (lr * grad_W3[:,np.newaxis])
                W2 = W2 - (lr * grad_W2[:,np.newaxis])
                W1 = W1 - (lr * grad_W1[:,np.newaxis])
            
            #y_train_predictions = x_train@W 
            #y_valid_predictions = x_valid@W
            #train_errors.append(mse(y_train,y_train_predictions[:,0]))
            #valid_errors.append(mse(y_valid,y_valid_predictions[:,0]))

        self.W1=W1
        self.W2=W2
        self.W3=W3
    
    def predict(self,X):
        
        X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
        n = X.shape[0]
        prediction = []
        
        for i in range(n):
            z1 = X[i,:]@self.W1
            z1 = z1[0]
            a1 = self.sigmoid(z1)
            
            z2 = X[i,:]@self.W2
            z2 = z2[0]
            a2 = self.sigmoid(z2)
            
            x1_2 = np.array([a1,a2,1])
            z3 = x1_2@self.W3
            z3 = z3[0]
            a3 = self.sigmoid(z3)
            
            prediction.append(a3)

        return np.array(prediction)
    
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = NeuralNetwork()
model.fit(X,y)

print(model.predict(X))