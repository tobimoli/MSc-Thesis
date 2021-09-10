# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:41:46 2021

@author: molenaar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA.csv'
data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
data['year'] = [int(n[6:10]) for n in data['time']]
data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
data['date'] = data['date'].astype("datetime64")

#remove outliers
data = data[data['chl']>0]

plt.figure(figsize = (6,6))
plt.scatter(data['dep'],data['chl'], marker = '.', alpha=0.05)
plt.xlabel('depth [m]')
plt.grid()
plt.ylabel('chl concentration [mg]')

z = np.polyfit(data['dep'],data['chl'], 20)
p = np.poly1d(z)
plt.plot(data['dep'],p(data['dep']),"r--")

plt.show()

plt.figure(figsize = (6,6))
plt.hist(data['dep'], bins=100)
plt.xlabel('depth [m]')
plt.ylabel('#observations')
plt.show()

plt.figure(figsize = (6,6))
plt.hist(data['chl'], bins=100)
plt.xlabel('chlorophyll [mg]')
plt.ylabel('#observations')
plt.show()

#%%
from sklearn.linear_model import LinearRegression

X = np.array([data['dep']]).T
reg = LinearRegression().fit(X, data['chl'])
print(reg.score(X, data['chl']))
x = reg.coef_
I = reg.intercept_

plt.figure(figsize = (6,6))
plt.scatter(data['dep'],data['chl'], marker = '.', alpha=0.05)
plt.plot([0,np.max(data['dep'])],[I,I+x*np.max(data['dep'])], 'r-')
plt.xlabel('depth [m]')
plt.grid()
plt.ylabel('chl concentration [mg]')
plt.show()


interval = 2
line = []
for i in range(int(np.max(data['dep']))-interval):
    line.append(np.mean(data['chl'][(data['dep']>i) & (data['dep']<i+interval)]))
    
plt.figure(figsize = (6,6))
plt.scatter(data['dep'],data['chl'], marker = '.', alpha=0.05)
plt.plot(list(range(int(np.max(data['dep']))-interval)),line, 'r-', label = "Naive Nonparametric Regression")
plt.xlabel('depth [m]')
plt.grid()
plt.legend()
plt.ylabel('chl concentration [mg]')
plt.show()

#%%
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
x = data['dep']
y = data['chl']
z = lowess(y, x, frac = 0.05)
plt.rcParams.update({'font.size': 18})

plt.figure(figsize = (8,8))
plt.scatter(data['chl'],data['dep'], marker = 'o', alpha=0.05, label = "Observations")
plt.plot(z[:,1],z[:,0], 'r-', label = "LOWESS, span = 0.05")
#plt.plot(w[:,0],w[:,1], 'g-', label = "LOWESS, span = 0.01")
plt.ylabel('Depth [m]')
plt.grid()
plt.gca().invert_yaxis()
plt.xlim([-.01,1.01])
leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel('Chl-a concentration [mg.m-3]')
plt.show()
