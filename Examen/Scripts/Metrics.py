# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:32:37 2021

@author: MaxiT
"""

import numpy as np

class BaseMetric:
    """
    Clase base métricas
    """
    def __init__(self):
        pass

    def __call__(self,truth, predicted):
        return NotImplemented
    
    
class MSE(BaseMetric):
    """
    Metrica MSE
    """
    def __init__(self):
        pass

    def __call__(self,truth, predicted):
        """
        Calculo de MSE
        
        Args:
            truth (numpy array): array de valores verdaderos
            predicted (numpy array): array de valores predichos

        Returns
            float: error cuadrático medio

        """
        mse= np.sum(np.power((predicted - truth),2))/ predicted.shape[0]
        return mse
    

class Precision(BaseMetric):
    """
    Precision metric
    """
    def __init__(self):
        pass

    def __call__(self, truth, prediction):
        TP = np.sum(truth & prediction)
        FP = np.sum((1-truth) & (prediction))
        precision= TP / (TP+FP)
        return precision

    
class Recall(BaseMetric):
    """
    Recall metric
    """
    def __init__(self):
        pass

    def __call__(self, truth, prediction):
        TP = np.sum(truth & prediction)
        FN = np.sum((truth) & (1-prediction))
        recall= TP / (TP+FN)
        return recall

    
class Accuracy(BaseMetric):
    """
    Accuracy metric
    """
    def __init__(self):
        pass

    def __call__(self, truth, prediction):
        TP = np.sum(truth & prediction)
        TN = np.sum((1-truth) & (1-prediction))
        FP = np.sum((1-truth) & (prediction))
        FN = np.sum((truth) & (1-prediction))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return accuracy
    
    

#metrics = [ ('MSE', MSE()) , \
#            ('Precision', Precision()) , \
#            ('Accuracy', Accuracy()) , \
#            ('Recall', Recall())
#          ]   
#y_truth = np.array([1,0,0,0,0,1,1,1,0,0])
#y_predicted = np.array([0,0,0,1,0,1,1,1,1,0])
#for metric in metrics:
#    print(metric[0] + ": {}".format(metric[1](y_truth,y_predicted)))