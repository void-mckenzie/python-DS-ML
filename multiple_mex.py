# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:25:03 2019

@author: Mukkesh McKenzie
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labs= LabelEncoder()
X[:,3]=labs.fit_transform(X[:,3])
hots= OneHotEncoder(categories='auto')
X = np.concatenate((X[:,0:3],hots.fit_transform(X[:,3].reshape(-1,1)).toarray()), axis=1)
#Dummy trap not necessary
X=np.concatenate((X[:,0:3],X[:,4:6]),axis=1)
X=X.astype(float)
X=np.concatenate((np.ones((len(y),1)).astype(float),X),axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

import statsmodels.formula.api as sm
#X=np.concatenate((np.ones((len(y),1)).astype(float),X),axis=1)
X_opt=X_train[:,[0,1,2,3,4,5]]
statreg=sm.OLS(endog=y_train,exog=X_train).fit()
statreg.summary()
statreg.pvalues
statreg.rsquared_adj
arr1=[0,1,2,3,4,5]
arrtemp=[]
tempr=0
while(1):
    X_opt=X_train[:,arr1]
    statreg=sm.OLS(endog=y_train,exog=X_opt).fit()
    print(statreg.summary())
    if(max(statreg.pvalues)<0.05):
        if(statreg.rsquared_adj<tempr):
            arr1=arrtemp
            X_opt=X_train[:,arr1]
        break
    else:
        tempr=statreg.rsquared_adj
        arrtemp=arr1
        m=max(statreg.pvalues)
        arr2=np.ones(len(arr1))
        temp=statreg.pvalues
        k=0
        for i in temp:
            if i==m:
                arr2[k]=0
                break
            else:
                k=k+1
        arrx=[]
        i=0
        while(i<len(arr1)):
            if(arr2[i]==0):
                i=i+1
            else:
                arrx.append(arr1[i])
                i=i+1
        arr1=arrx
statreg=sm.OLS(endog=y_train,exog=X_opt).fit()

regressor1 = LinearRegression()
regressor1.fit(X_opt,y_train)

y_pred1=regressor1.predict(X_test[:,arr1])



plt.scatter(y_test,y_pred)
plt.scatter(y_test,y_pred1,color="red")
plt.plot(y_test,y_test,color="green")

