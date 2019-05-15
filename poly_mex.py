import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg1=LinearRegression()
linreg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

linreg1.fit(X_poly,y)

plt.scatter(X,y)
plt.plot(X,linreg.predict(X))
plt.scatter([6.5],linreg.predict([[6.5]]),color='red')

plt.scatter(X,y)
plt.plot(X,linreg1.predict(X_poly))
plt.scatter([6.5],linreg1.predict(poly_reg.fit_transform([[6.5]])),color='green')