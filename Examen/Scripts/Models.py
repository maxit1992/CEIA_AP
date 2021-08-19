# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:51:29 2021

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
    
    
    
class PCAModel(BaseModel):
    
    def __init__(self):
        pass
    
    def fit_transform(self, X, n_components=2):
        X = X - X.mean(axis=0)
        cov = np.cov(X.T)
        # v son los autovalores y w los autovectores
        v, w = np.linalg.eig(cov)
        idx = v.argsort()[::-1]
        v = v[idx]
        w = w[:, idx]
        self.autovalues= v[:n_components]
        self.model = w[:, :n_components]
        return X.dot(self.model)
    
    def fit(self, X):
        return NotImplemented
    
    def transform(self,X):
        return NotImplemented
    
    def predict(self,X):
        return NotImplemented
    
    

class LinearRegressionModel(BaseModel):
    """
    Modelo de aproximación con regresión afín a la lineal
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y, bias=True):
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
            
        y=y[:,np.newaxis]
        self.model= np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
        self.bias = bias
    
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
    
    
class PolinomicRegressionModel(BaseModel):
    """
    Modelo de aproximación con regresión afín a la lineal
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y, deg=1, bias=True):
        """
        Función de ajuste de W
        
        Args:
            x (numpy array): array de valores de entrada
            y (numpy array): array de valores de salida

        """
        if len(X.shape)<2:
            X=X[:,np.newaxis]
            
        X_aux = X.copy()
        for i in range(2,deg+1):
            X=np.hstack([X,X_aux**i])
            
        if bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
            
        y=y[:,np.newaxis]
        self.model= np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
        self.bias = bias
        self.deg=deg
    
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
            
        X_aux = X.copy()
        for i in range(2,self.deg+1):
            X=np.hstack([X,X_aux**i])

        if self.bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)
            
        y=X@self.model
        return y[:,0]


class LogisticRegressionModel(BaseModel):
    
    model = None
    bias = True
    loss_list = []
    
    def sigmoid(self, x):
        g_x = 1 / (1 + np.exp(-x))
        return g_x
    
    def cost_function (self, y_real, y_predicted):
        j_w=- np.mean( y_real * np.log(y_predicted) + \
                      (1.0-y_real) * np.log((1.0-y_predicted)))
        return j_w
    
    def gradient(self, x,y_real,y_predicted):
        error = (y_predicted-y_real)
        dj_w = np.sum(error[:,np.newaxis]*x,axis=0) / y_real.shape[0]
        return dj_w

    def fit(self, X, y, lr=0.1, b=10, epochs=1000, bias=True, verbose=False):
        if len(X.shape)<2:
            X=X[:,np.newaxis]
        if bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)

        m = X.shape[1]

        # initialize random weights
        W = np.zeros(shape=(m, 1),dtype='float64')

        # iterate over the n_epochs
        for j in range(1,epochs+1):
            # Shuffle all the samples 
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

            # Calculate the batch size in samples as a function of the number of batches
            batch_size = int(len(X) / b)

            # Iterate over the batches
            for i in range(0, len(X), batch_size):

                end = i + batch_size if i + batch_size <= len(X) else len(X)
                batch_x = X[i: end] # batch_size*m
                batch_y = y[i: end] # batch_size*1

                # Calculate the prediction for the whole batch
                prediction = self.sigmoid(batch_x@W)  # batch_sizex1
                prediction=prediction[:,0]

                # Calculate the gradient for the batch
                grad = self.gradient(batch_x, batch_y, prediction)

                # Update the weights
                W = W - (lr * grad[:,np.newaxis])

                # Calculate new loss
                loss= self.cost_function(y, self.sigmoid(X@W)[:,0])
                self.loss_list.append(loss)
                
            if j>0 and j % 10==0 and verbose:
                print ('Loss epoch {}'.format(j) + ': {}'.format(loss))

        self.model = W
        self.bias = bias
        
    def predict(self, X):
        if len(X.shape)<2:
            X=X[:,np.newaxis]

        if self.bias:
            X=np.append(X, np.ones(shape=(X.shape[0],1)), axis = 1)

        prediction = self.sigmoid(X@self.model)  # batch_sizex1
        prediction = prediction[:,0]
        prediction = (prediction>=0.5) * 1

        return prediction
    
#x = np.array([[0.8, 0.7], [0.1, -0.1],[0.5, -0.5]])
#pca = PCAModel()
#x_pca_np=pca.fit_transform(x,1)

#x = np.array([[0.8, 0.7, 0.4, 0.2], [0.2,0.4,0.5,0.6],[999,251,4089,9999]])
#x = x.T
#y = 3 * x[:,0] + x[:,1]
#linear_model = LinearRegressionModel()
#linear_model.fit(x,y,False)
#y_pred = linear_model.predict(x)

#x = np.array([0.8, 0.7, 0.4, 0.2])
#y = 3 * x + 4 *  x**2 + 5
#pol_model= PolinomicRegressionModel()
#pol_model.fit(x,y,2,True)
#y_pred = pol_model.predict(x)

#x = np.array([0.8, 0.7, 0.4, 0.2, 0.5, 0.49, 0.3, 0.0])
#y = (x>=0.7)*1.0
#logistic_model= LogisticRegressionModel()
#logistic_model.fit(x,y,0.5,2,50000, True, True)
#y_pred = logistic_model.predict(x)

