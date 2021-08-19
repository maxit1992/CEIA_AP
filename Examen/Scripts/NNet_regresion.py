# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:46:41 2021

@author: MaxiT
"""
import numpy as np
from sklearn import model_selection
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, X, Y):
      self.X=X
      self.Y=Y
    
    def __len__(self):
      return self.X.shape[0]
    
    def __getitem__(self, idx):
      return self.X[idx,:], self.Y[idx]
  

class TestCustomDataset(Dataset):
    def __init__(self, X):
      self.X=X
    
    def __len__(self):
      return self.X.shape[0]
    
    def __getitem__(self, idx):
      return self.X[idx,:]


class NNetLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features=2, out_features=10, bias = True)
        self.activation_1 = torch.nn.ReLU()
        self.dropout_1= torch.nn.Dropout(p=0.05)
        self.linear_2 = torch.nn.Linear(in_features=10, out_features=20, bias = True)
        self.activation_2 = torch.nn.ReLU()
        self.dropout_2= torch.nn.Dropout(p=0.05)
        self.linear_3 = torch.nn.Linear(in_features=20, out_features=1, bias = True)

    def forward(self, x):
        # X es el batch que va a entrar
        z1 = self.linear_1(x)
        a1 = self.activation_1(z1)
        d1 = self.dropout_1(a1)
        z2 = self.linear_2(d1)
        a2 = self.activation_2(z2)
        d2 = self.dropout_2(a2)
        y = self.linear_3(d2)
        return y
  
    
class NnetRegresion():
    
    def __init__(self):
        self.nnet = NNetLayers()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass
        
    def metrics_se(self,truth, predicted):
        mse= np.sum(np.power((predicted - truth),2))
        return mse
        
    def fit(self,x_train, y_train, x_valid=None, y_valid=None, \
            batch_size = 32, lr=0.001, epochs=100, verbose=True):
        
        training_set = CustomDataset(x_train, y_train)
        training_dataloader = DataLoader(training_set,batch_size=batch_size, \
                                         shuffle=True)
        
        if (x_valid is not None) & (y_valid is not None):
            valid_set = CustomDataset(x_valid, y_valid)     
            valid_dataloader = DataLoader(valid_set,batch_size=len(valid_set), \
                                          shuffle= True)
        
        # Optimizer
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(self.nnet.parameters(),\
                                    lr=0.001)
    
        # Training
        self.nnet.to(self.device)
        
        history_loss=[]
        history_valid_mse=[]
        
        for epoch in range(epochs):
            running_loss = 0
            train_mse = 0
            valid_mse = 0
            self.nnet.train()
            for i, data in enumerate(training_dataloader):
                # data es una tupla batch (data, label)
                x, y = data #todavia esta en numpy
                x = x.to(self.device).float() #convierte a tensores y pasa a GPU si esta disponible
                y = y.to(self.device).float() #convierte a tensores y pasa a GPU si esta disponible
    
                # set gradient to zero
                optimizer.zero_grad()
    
                # forward
                y_hat = self.nnet(x)
    
                # loss
                loss = criterion(y_hat[:,0], y)
    
                # backward
                loss.backward()
    
                # update of parameters
                optimizer.step()
    
                # compute loss and statistics
                running_loss += loss.item()
                train_mse += self.metrics_se(y.detach().numpy(),\
                                             y_hat[:,0].detach().numpy())
                
                history_loss.append(running_loss/x_train.shape[0])
            
            if (verbose) & ((epoch) % (epochs/10)==0):
                self.nnet.eval()
                with torch.no_grad():
                        
                    if (x_valid is not None) & (y_valid is not None):
                        for i, data in enumerate(valid_dataloader):
                            # batch
                            x, y = data
                            x = x.to(self.device).float()
                            y = y.to(self.device).float()
                
                            # forward 
                            y_hat = self.nnet(x)
                
                            # accumulate data
                            valid_mse += self.metrics_se(y.detach().numpy(),\
                                                         y_hat[:,0].detach().numpy())
                                
                            history_valid_mse.append(valid_mse/x_valid.shape[0])
            
                        print(f"Epoch = {epoch} | " + \
                              f"loss = {running_loss / x_train.shape[0]} | " +\
                                  f"valid mse: {valid_mse/x_valid.shape[0]}")
                    else:
                        print(f"Epoch = {epoch} | " + \
                              f"loss = {running_loss / x_train.shape[0]}")
                            
        return history_loss,history_valid_mse
        
    def predict(self,x):
        
        self.nnet.eval()
        with torch.no_grad():
            test_set = TestCustomDataset(x)
            test_dataloader = DataLoader(test_set,batch_size=len(test_set), \
                                         shuffle= False)
            
            for i, data in enumerate(test_dataloader):
                x = data 
                x = x.to(self.device).float() 
                y_hat = self.nnet(x)

        return y_hat[:,0].detach().numpy()
                    


def test_NnetRegresion(): 
    x = np.random.uniform(-10,10,size=10000)
    Y = np.power(x-2,2) + 3 + np.random.normal(0, 0.1, 10000)
    x=x[:,np.newaxis]
    x=np.append(x**2, x, axis = 1)
    
    plt.figure()
    plt.scatter(x[:,1],Y)
    plt.title('Dataset Train')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    x_train, x_test, Y_train, Y_test = \
        model_selection.train_test_split( x, Y, test_size=0.2, random_state=5)
        
    x_train, x_valid, Y_train, Y_valid= \
        model_selection.train_test_split( x_train, Y_train, \
                                         test_size=0.2, random_state=5)
            
    model = NnetRegresion()
    
    model.fit(x_train,Y_train,x_valid,Y_valid)
    
    y_test_hat = model.predict(x_test)
    
    plt.figure()
    plt.scatter(x_test[:,1],Y_test)
    idx_sort = np.argsort(x_test[:,1])
    plt.plot(x_test[idx_sort,1],y_test_hat[idx_sort],color='red')
    plt.title('Dataset Test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Run test
test_NnetRegresion()