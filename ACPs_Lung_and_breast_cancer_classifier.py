import pandas as pd
import numpy as np

#%% Load or Import the dataset
path='E:\BS-CS-Vl\Machine Leaining Projects\Anticancer_panides\ACPs_Breast_cancer.csv'
path1='E:\BS-CS-Vl\Machine Leaining Projects\Anticancer_panides\ACPs_Lung_cancer.csv'
breast_df=pd.read_csv(path) 
Lung_df=pd.read_csv(path1) 

#%% Removing Unwanted features
breast_df=breast_df.drop('ID',axis=1)
Lung_df=Lung_df.drop('ID',axis=1)
#%% Encode the data

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
arr=['sequence','class']
for i in arr:
    breast_df[i]=le.fit_transform(breast_df[i])
    Lung_df[i]=le.fit_transform(Lung_df[i])
#%% Seperate X and y features

breast_x=breast_df.values[:,:-1]
breast_y=breast_df.values[:,np.newaxis,-1]
Lung_x=breast_df.values[:,:-1]
Lung_y=breast_df.values[:,np.newaxis,-1]

#%% Seperate training and testing data
def splitdata(X,y):
    r=0.2
    ln=len(X)
    ts_Ins=round(ln*r)
    tr_x = X[:-ts_Ins,:]
    ts_x = X[-ts_Ins:,:]
    tr_y = y[:-ts_Ins]
    ts_y = y[-ts_Ins:]
    return tr_x,ts_x,tr_y,ts_y

breast_tr_x,breast_ts_x,breast_tr_y,breast_ts_y=splitdata(breast_x,breast_y)
Lung_tr_x,Lung_ts_x,Lung_tr_y,Lung_ts_y=splitdata(Lung_x,Lung_y)


#%%Train the model with logistix regression
from sklearn.linear_model import LogisticRegression
logreg_breast = LogisticRegression()
logreg_breast.fit(breast_tr_x,breast_tr_y)

logreg_Lung = LogisticRegression()
logreg_Lung.fit(Lung_tr_x,Lung_tr_y)

#%% Testing
breast_pred = logreg_breast.predict(breast_ts_x)
Lung_pred = logreg_Lung.predict(Lung_ts_x)

#%%Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(breast_ts_y, breast_pred)
print("Accuracy of breast cancer model : {:.2f}%".format(accuracy * 100))

accuracy = accuracy_score(Lung_ts_y, Lung_pred)
print("Accuracy of Lung cancer model : {:.2f}%".format(accuracy * 100))

