# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:59:29 2021

@author: MaxiT
"""

import numpy as np

class BaseModel():
    """
    Base class for models
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        return NotImplemented
    
    def transform(self,X):
        return NotImplemented
    
    def fit_transform(self, X, y):
        return NotImplemented
    
    def predict(self, X):
        return NotImplemented
    
    
class StochasticGradientDescentModel(BaseModel):
    """
    Modelo de aproximación con regresión afín a la lineal
    """
    model = None
    bias = None
    
    def __init__(self):
        pass
    
    def fit(self, X, y, bias=True, lr=0.1, epochs=1000):
        """
        Función de ajuste de W
        
        Args:
            x (numpy array): array de valores de entrada
            y (numpy array): array de valores de salida

        """
        if len(X.shape)<2:
            X=X[:,np.newaxis]
            
        if bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
    
        n = X.shape[0]
        m = X.shape[1]

        # initialize random weights
        W = np.zeros(shape=(m, 1),dtype='float64') # mx1

        # Iterate over the n_epochs
        for i in range(epochs):
        
            # Shuffle all the samples
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

            # Iterate over each sample
            for j in range(n):
                # Calculate the prediction for each sample
                prediction = X[j]@W  # nx1
                # Calculate the error for each sample
                error = y[j] - prediction  # 1x1
                # Calculate the gradient for each sample
                grad_sum = error * X[j] # 1xm
                grad_mul = -2/n * grad_sum  # 1xm
                gradient = grad_mul[:,np.newaxis]
                
                # Update all the weights
                W = W - (lr * gradient) # mx1

        self.model=W
        self.bias = bias
        return
    
    def predict(self,X):
        """
        Función de predicción
        
        Args:
            x (numpy array): array de valores de entrada
            
        Returns
            numpy array: array de predicciones

        """
        if len(X.shape)<2:
            X=X[:,np.newaxis]
            
        if self.bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
            
        y=X@self.model
        return y[:,0]
    

class BatchGradientDescentModel(BaseModel):
    """
    Modelo de aproximación con regresión afín a la lineal
    """
    model = None
    bias = None
    
    def __init__(self):
        pass
    
    def fit(self, X, y, bias=True, lr=0.1, epochs=1000, b=2):
        """
        Función de ajuste de W
        
        Args:
            x (numpy array): array de valores de entrada
            y (numpy array): array de valores de salida

        """
        if len(X.shape)<2:
            X=X[:,np.newaxis]
        
        if bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
    
        m = X.shape[1]

        # initialize random weights
        W = np.zeros(shape=(m, 1),dtype='float64')

        # iterate over the n_epochs
        for i in range(epochs):

            # Shuffle all the samples 
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

            # Calculate the batch size in samples as a function of the number of batches
            batch_size = int(len(X) / b)

            # Iterate over the batches
            for i in range(0, len(X), batch_size):

                end = i + batch_size if i + batch_size <= len(X) else len(X)
                batch_X = X[i: end] # batch_size*m
                batch_y = y[i: end] # batch_size*1

                # Calculate the prediction for the whole batch
                prediction = batch_X@W  # batch_sizex1
                # Calculate the error for the whole batch
                error = batch_y - prediction[:,0]  # batch_sizex1

                # Calculate the gradient for the batch

                # error[batch_sizex1]*batch_X[batch_size*m]--> broadcasting --> batch_size*m
                grad_sum = np.sum(error[:,np.newaxis]* batch_X, axis=0) # 1xm
                grad_mul = -2/batch_size * grad_sum  # 1xm
                gradient = grad_mul[:,np.newaxis]  # mx1

                # Update the weights
                W = W - (lr * gradient)

        self.model=W
        self.bias=bias
        return
    
    def predict(self,X):
        """
        Función de predicción
        
        Args:
            x (numpy array): array de valores de entrada
            
        Returns
            numpy array: array de predicciones

        """
        if len(X.shape)<2:
            X=X[:,np.newaxis]
            
        if self.bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
            
        y=X@self.model
        return y[:,0]

#z = np.array([[0.8, 0.7, 0.4, 0.2], [0.2,0.4,0.5,0.6]])
#z = z.T
#y = 3 * z[:,0] + z[:,1]
#linear_model = StochasticGradientDescentModel()
#linear_model = BatchGradientDescentModel()
#linear_model.fit(z,y,True)
#y_pred = linear_model.predict(z)