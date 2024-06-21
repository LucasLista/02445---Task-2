import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# Import HR_data into a DataFrame
HR_data = pd.read_csv('HR_data.csv')
HR_data.head()
scaler = StandardScaler()
scaler.fit(HR_data[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']])
Scaled_data = scaler.transform(HR_data[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']])
inputcols = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
Scaled_data = pd.DataFrame(Scaled_data, columns=inputcols)

def find_best_logistic(Scaled_data, Frustrated):
    CV = KFold(n_splits=10, shuffle=True, random_state=42)
    errors = np.empty((100,2,10))
    cv = 0
    for train_index, test_index in CV.split(Scaled_data):
        i=0
        X_train, X_test, y_train, y_test = Scaled_data[train_index], Scaled_data[test_index], Frustrated[train_index], Frustrated[test_index]
        for C in np.logspace(-3, 3, 100):
            logmodel = LogisticRegression(C=C, penalty='l2',max_iter = 10**6)
            logmodel.fit(X_train, y_train)
            errors[i,:,cv]= C, np.sqrt(mean_squared_error(logmodel.predict(X_test), y_test))
            i += 1
        cv += 1
    errors = np.mean(errors, axis=2)
    best= np.argmin(errors[:,1])
    return errors[best,0]

def find_best_KNN(Scaled_data, Frustrated):
    CV = KFold(n_splits=10, shuffle=True, random_state=42)
    max_n = int(len(Frustrated)*9/10-1)
    errors = np.empty((max_n,2,10))
    cv = 0
    for train_index, test_index in CV.split(Scaled_data):
        i=0
        X_train, X_test, y_train, y_test = Scaled_data[train_index], Scaled_data[test_index], Frustrated[train_index], Frustrated[test_index]
        for n in range(1, max_n+1):
            knn_model = KNeighborsRegressor(n_neighbors=n)
            knn_model.fit(X_train, y_train)
            errors[i,:,cv]= n, np.sqrt(mean_squared_error(knn_model.predict(X_test), y_test))
            i += 1
        cv += 1
    errors = np.mean(errors, axis=2)
    plt.plot(errors[:,0],errors[:,1])
    best = np.argmin(errors[:,1])
    return errors[best,0]
        
def find_best_RF(Scaled_data, Frustrated):
    CV = KFold(n_splits=10, shuffle=True, random_state=42)
    errors = np.empty((50,2,10))
    cv = 0
    for train_index, test_index in CV.split(Scaled_data):
        i=0
        X_train, X_test, y_train, y_test = Scaled_data[train_index], Scaled_data[test_index], Frustrated[train_index], Frustrated[test_index]
        for n in range(1, 250, 5):
            RFmodel = RandomForestRegressor(n_estimators = n, random_state=42)
            RFmodel.fit(X_train, y_train)
            errors[i,:,cv]= n, np.sqrt(mean_squared_error(RFmodel.predict(X_test), y_test))
            i += 1
        cv += 1
    errors = np.mean(errors, axis=2)
    plt.plot(errors[:,0],errors[:,1])
    best = np.argmin(errors[:,1])
    return errors[best,0]

def find_best_Ridge(Scaled_data, Frustrated):
    CV = KFold(n_splits=10, shuffle=True, random_state=42)
    errors = np.empty((1000,2,10))
    cv = 0
    for train_index, test_index in CV.split(Scaled_data):
        i=0
        X_train, X_test, y_train, y_test = Scaled_data[train_index], Scaled_data[test_index], Frustrated[train_index], Frustrated[test_index]
        for a in np.logspace(-3,6,1000):
            Ridge_model = Ridge(alpha = a)
            Ridge_model.fit(X_train, y_train)
            errors[i,:,cv]= a, np.sqrt(mean_squared_error(Ridge_model.predict(X_test), y_test))
            i += 1
        cv += 1
    errors = np.mean(errors, axis=2)
    plt.semilogx(errors[:,0],errors[:,1])
    best = np.argmin(errors[:,1])
    return errors[best,0]

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(6, 512))
        
        # Hidden layers
        self.layers.append(nn.Linear(512, 256))
        self.layers.append(nn.Linear(256, 128))
        self.layers.append(nn.Linear(128, 64))

        # Output layer
        self.layers.append(nn.Linear(64, 1))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        # Apply the last layer without ReLU
        x = self.layers[-1](x)
        return x.reshape(-1)
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

def find_best_ANN(Scaled_data, Frustrated):
    device = 'cpu'
    criterion = nn.MSELoss()
    verbose = False

    X_train, X_test, y_train, y_test = train_test_split(Scaled_data, Frustrated, test_size=0.1)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    # Use GPU if available
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)


    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
            
    model = ANNModel()
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.01)    

    # For loop for training the model 10 times and choosing the best one
    best_loss = -1
    best_epoch = 0
    best_running_loss = -1
    for i in range(3):

        # Training the model
        running_loss = []
        epoch = 0
        local_best_loss = -1
        local_best_epoch = 0
        while epoch-local_best_epoch <= 200:
            model.train()
            total_train_loss = 0
            for inputs, labels in train_loader:
                # inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_train_loss += loss.item()*len(labels)
                loss.backward()
                optimizer.step()
                
            if epoch % 1 == 0:
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    total_loss = 0
                    for inputs, labels in test_loader:
                        # inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()*len(labels)
                    test_loss = total_loss / len(test_dataset)
                    if len(running_loss) < 5:
                        running_loss.append(test_loss)
                    else:
                        running_loss.pop(0)
                        running_loss.append(test_loss)
                    if epoch == 0 or epoch % 10 == 9:
                        print(f'Epoch: {epoch+1}, Test Loss: {test_loss}, Train Loss: {total_train_loss/len(train_dataset)}') if verbose else None
                    if best_loss == -1 or (best_loss >= test_loss and best_running_loss >= np.mean(running_loss)):
                        best_epoch = epoch
                        best_loss = test_loss
                        best_running_loss = np.mean(running_loss)
                        bismodel=copy.deepcopy(model.state_dict())
                    if local_best_loss == -1 or local_best_loss >= test_loss:
                        local_best_loss = test_loss
                        local_best_epoch = epoch
            epoch += 1
    return bismodel
