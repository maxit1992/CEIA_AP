# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:46:41 2021

@author: MaxiT
"""
import numpy as np
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure()
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    


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
        self.linear_3 = torch.nn.Linear(in_features=20, out_features=4, bias = True)

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
  
    
class NnetMultiClass():
    
    def __init__(self):
        self.nnet = NNetLayers()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass
    
        
    def fit(self,x_train, y_train, x_valid=None, y_valid=None, \
            batch_size = 32, lr=0.001, epochs=50, verbose=True):
        
        
        self.y_encoder = OneHotEncoder(sparse=True)
        self.y_encoder.fit(y_train.reshape(-1, 1))        
        
        training_set = CustomDataset(x_train, y_train)
        training_dataloader = DataLoader(training_set,batch_size=batch_size, \
                                         shuffle=True)
        
        if (x_valid is not None) & (y_valid is not None):
            valid_set = CustomDataset(x_valid, y_valid)     
            valid_dataloader = DataLoader(valid_set,batch_size=len(valid_set), \
                                          shuffle= True)
        
        # Optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(self.nnet.parameters(),\
                                    lr=0.001)
    
        # Training
        self.nnet.to(self.device)
        
        history_loss=[]
        history_train_auc=[]
        history_valid_auc=[]
        
        for epoch in range(epochs):
            running_loss = 0
            nnet_train_scores = []
            train_truth = []
            self.nnet.train()
            for i, data in enumerate(training_dataloader):
                x, y = data 
                x = x.to(self.device).float() 
                #y = y.detach().numpy()
                #y = self.y_encoder.transform(y.reshape(-1, 1)).toarray()
                #y = torch.Tensor(y)
                y = y.to(self.device).long() 
    
                # set gradient to zero
                optimizer.zero_grad()
    
                # forward
                y_hat = self.nnet(x)
    
                # loss
                loss = criterion(y_hat, y)
    
                # backward
                loss.backward()
    
                # update of parameters
                optimizer.step()
    
                # compute loss and statistics
                running_loss += loss.item()
                
                y_hat = torch.softmax(y_hat,1)
                
                y_true = y.detach().numpy()
                y_true = self.y_encoder.transform(y_true.reshape(-1, 1)).toarray()
                train_truth += list(y_true) 
                nnet_train_scores += list(y_hat.detach().numpy())
                
            history_loss.append(running_loss/x_train.shape[0])
            
            train_auc = roc_auc_score(train_truth, nnet_train_scores,\
                                      multi_class='ovr')
            history_train_auc.append(train_auc)
    
            
            if (verbose) & ((epoch) % (epochs/10)==0):
                self.nnet.eval()
                with torch.no_grad():
                        
                    if (x_valid is not None) & (y_valid is not None):
                        
                        nnet_valid_scores = []
                        valid_truth = []
                    
                        for i, data in enumerate(valid_dataloader):
                            # batch
                            x, y = data
                            x = x.to(self.device).float()
                            #y = y.detach().numpy()
                            #y = self.y_encoder.transform(y.reshape(-1, 1)).toarray()
                            #y = torch.Tensor(y)
                            y = y.to(self.device).float()
                
                            # forward 
                            y_hat = self.nnet(x)
                
                            y_hat = torch.softmax(y_hat,1)
                
                            y_true = y.detach().numpy()
                            y_true = self.y_encoder.transform(y_true.reshape(-1, 1)).toarray()
                            valid_truth += list(y_true) 
                            nnet_valid_scores += list(y_hat.detach().numpy())
                               
                        valid_auc = roc_auc_score(valid_truth, nnet_valid_scores, \
                                                   multi_class='ovr')
                        history_valid_auc.append(valid_auc)
            
                        print(f"Epoch = {epoch} | " + \
                              f"loss = {running_loss / x_train.shape[0]} | " + \
                                  f"train auc: {train_auc}" + \
                                  f"valid auc: {valid_auc}")
                    else:
                        print(f"Epoch = {epoch} | " + \
                              f"loss = {running_loss / x_train.shape[0]} | " +\
                                   f"train auc: {train_auc}")
                            
        return history_loss,history_train_auc, history_valid_auc
        
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
                y_hat = torch.softmax(y_hat,1)
                y_hat = y_hat.detach().numpy()
                y_hat = self.y_encoder.inverse_transform(y_hat)
        return y_hat[:,0]
    
    def predict_proba(self,x):
        self.nnet.eval()
        with torch.no_grad():
            test_set = TestCustomDataset(x)
            test_dataloader = DataLoader(test_set,batch_size=len(test_set), \
                                         shuffle= False)
            
            for i, data in enumerate(test_dataloader):
                x = data 
                x = x.to(self.device).float() 
                y_hat = self.nnet(x)
                y_hat = torch.softmax(y_hat,1)
                y_hat = y_hat.detach().numpy()

        return y_hat
                    


def test_NnetMultiClass(): 
    
    X1 = np.random.uniform(0,8,10000)
    U = np.random.uniform(0,4,10000)
    N1 = np.random.normal(3,0.1,10000)
    N2 = np.random.normal(-1,0.1,10000)
    N3 = np.random.normal(5,0.1,10000)
    N4 = np.random.normal(-5,0.1,10000)
    
    X2 = (X1-4)**2
    
    X2[U<1] = X2[U<1] + N1[U<1]
    X2[(U>=1) & (U<2)] = X2[(U>=1) & (U<2)] + N2[(U>=1) & (U<2)]
    X2[(U>=2) & (U<3)] = X2[(U>=2) & (U<3)] + N3[(U>=2) & (U<3)]
    X2[(U>=3) & (U<4)] = X2[(U>=3) & (U<4)] + N4[(U>=3) & (U<4)]
    
    Y = np.zeros(10000)
    
    Y[(U<1)] = 0
    Y[(U>=1) & (U<2)] = 1
    Y[(U>=2) & (U<3)] = 2
    Y[(U>=3) & (U<4)] = 3
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(X1[Y==0],X2[Y==0], color='blue')
    ax.scatter(X1[Y==1],X2[Y==1], color='red')
    ax.scatter(X1[Y==2],X2[Y==2], color='green')
    ax.scatter(X1[Y==3],X2[Y==3], color='yellow')
    plt.show()
    
    X1=X1[:,np.newaxis]
    X2=X2[:,np.newaxis]
    x=np.append(X1, X2, axis = 1)
    
    x_train, x_test, Y_train, Y_test = \
        model_selection.train_test_split( x, Y, test_size=0.2, random_state=5)
        
    x_train, x_valid, Y_train, Y_valid= \
        model_selection.train_test_split( x_train, Y_train, \
                                         test_size=0.2, random_state=5)
            
    model = NnetMultiClass()
    
    model.fit(x_train,Y_train,x_valid,Y_valid)
    
    y_test_hat = model.predict(x_test)
    
    plt.figure()
    plt.scatter(x_test[y_test_hat==0,0],x_test[y_test_hat==0,1],color='blue')
    plt.scatter(x_test[y_test_hat==1,0],x_test[y_test_hat==1,1],color='red')
    plt.scatter(x_test[y_test_hat==2,0],x_test[y_test_hat==2,1],color='green')
    plt.scatter(x_test[y_test_hat==3,0],x_test[y_test_hat==3,1],color='yellow')
    plt.title('Dataset Test')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    test_accuracy = accuracy_score(Y_test,y_test_hat)
    test_recall = recall_score(Y_test,y_test_hat,average='macro')
    test_precision = precision_score(Y_test, y_test_hat,average='macro')
    test_f1 = f1_score(Y_test,y_test_hat, average='macro')
    print(f"Accuracy: {test_accuracy}")
    print(f"Recall: {test_recall}")
    print(f"Precision: {test_precision}")
    print(f"F1-Score: {test_f1}")
    
    conf_matrix = confusion_matrix(Y_test, y_test_hat)
    plot_confusion_matrix(conf_matrix,target_names = np.unique(Y_test), \
                          title = "Confusion Matrix")
#Run test
test_NnetMultiClass() 