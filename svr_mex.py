# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:38:15 2019

@author: Mukkesh McKenzie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y= sc_y.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(X,y)
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

yolo=np.arange(float(min(X)),float(max(X)),0.1).reshape(-1,1)
plt.scatter(X,y)
plt.plot(yolo,regressor.predict(yolo))

