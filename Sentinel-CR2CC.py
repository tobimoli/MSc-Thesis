# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:25:00 2021

@author: molenaar
"""

#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from Function_file import Data, simple_idw, plotgrid
#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled_C2RCC"

b = Data(folder, name)
b.import_data()

#%% select area
#Lo1, Lo2, La1, La2 = 24.8,25.20,40.40,40.70 #cloud
#Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #full map
Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map
#Lo1, Lo2, La1, La2 = 25, 25.40,39.75,40.05 #limnos

lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)
coord = [lo1, lo2, la1, la2]

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
chl = b.var_values['conc_chl'][la1:la2,lo1:lo2].flatten()
unc = b.var_values['unc_chl'][la1:la2,lo1:lo2].flatten()
iop = b.var_values['iop_apig'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame(np.array([lon,lat,chl,unc,iop]).T)
x_sat.columns = ['lon','lat','chl','unc', 'iop']

#%% Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA.csv'

data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
data['year'] = [int(n[6:10]) for n in data['time']]
data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#remove outliers
data = data[data['chl']>0]

x_situ = data[['lon', 'lat', 'dep', 'chl']]
X = pd.concat([x_sat, x_situ], ignore_index=True, sort=False)
print(x_sat.shape)

#%%GPR for finding reflectances

iop_sat = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat['iop']), np.array(x_situ['lon']), np.array(x_situ['lat']))
unc_sat = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat['unc']), np.array(x_situ['lon']), np.array(x_situ['lat']))

#chl_sat = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['chl', 'unc']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
chl_situ = np.array(x_situ['chl'])
depth = x_situ['dep']<10

df = pd.DataFrame(np.array([iop_sat, chl_situ, depth]).T, columns = ('iop', 'chl', 'depth'))

plt.figure(figsize = (8,5))
ax = sns.scatterplot(x='iop', y='chl', data = df, hue = 'depth')
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['>5 meter', '<5 meter'])

#%% Non-linear least squares ALL DATA
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
chl_situ = np.array(x_situ['chl'])
def fun(x, a, b):
    y = a*x**b
    return y

K = 10; i = 0
kf = KFold(n_splits=K, shuffle = True, random_state = 0)
mse = np.zeros(K)
y_hat = np.zeros((len(chl_situ)))


start = time.time()
for train, test in kf.split(chl_situ):
    print(i)
    factor1, exponent1 = curve_fit(fun, xdata = iop_sat[train], ydata = chl_situ[train])[0]

    y_hat[test] = fun(iop_sat[test], factor1, exponent1)
    mse[i] = np.mean((chl_situ[test]-y_hat[test])**2)
    i += 1
    
print("The MSE (s.e.) = " + str(np.round(np.mean(mse),5)) + ' (' + str(np.round(mse.std()/np.sqrt(K),5)) + ')')
print("R-squared = " + str(1 - np.sum((y_hat - chl_situ)**2)/np.sum((chl_situ - np.mean(chl_situ))**2)))
print("AIC = " + str(len(chl_situ)*np.log(np.mean(mse)) + 2*(1+2)))
print("BIC = " + str(len(chl_situ)*np.log(np.mean(mse)) + np.log(len(chl_situ))*(1+2)))

#%%
np.random.seed(101)
#chl_situ = chl_situ.flatten() + np.random.normal(0, 0.005, size = len(chl_situ))
plt.figure(figsize = (5,5))
#plt.plot([0,.5],[0,.5], '-'); plt.plot(Prediction[0], Y_train, '.', markersize = 6)
plt.xlim([0.08, 0.11])
plt.ylim([0, 0.5])
plt.plot([0,0.5],[0,0.5], '-'); plt.plot(y_hat, chl_situ, '.', markersize = 6)
plt.xlabel("Prediction CHL-a conc."); plt.ylabel("Observed CHL-a conc."); plt.title("C2RCC (all)")
plt.show()
#%% Non-linear least squares DATA ONLY DEPTH <5 meters
chl_situ = np.array(x_situ['chl'])

K = 10; i = 0
kf = KFold(n_splits=K, shuffle = True, random_state = 0)
mse = np.zeros(K)
y_hat = np.zeros((len(chl_situ[depth])))


start = time.time()
for train, test in kf.split(chl_situ[depth]):
    print(i)
    factor1, exponent1 = curve_fit(fun, xdata = iop_sat[depth][train], ydata = chl_situ[depth][train])[0]
    
    y_hat[test] = fun(iop_sat[depth][test], factor1, exponent1)
    mse[i] = np.mean((chl_situ[depth][test]-y_hat[test])**2)
    i += 1
    
print("The MSE (s.e.) = " + str(np.round(np.mean(mse),5)) + ' (' + str(np.round(mse.std()/np.sqrt(K),5)) + ')')
print("R-squared = " + str(1 - np.sum((y_hat - chl_situ[depth])**2)/np.sum((chl_situ[depth] - np.mean(chl_situ[depth]))**2)))
print("AIC = " + str(len(chl_situ[depth])*np.log(np.mean(mse)) + 2*(1+2)))
print("BIC = " + str(len(chl_situ[depth])*np.log(np.mean(mse)) + np.log(len(chl_situ[depth]))*(1+2)))
np.random.seed(101)
chl_situ = chl_situ + np.random.normal(0, 0.005, size = len(chl_situ))
plt.figure(figsize = (5,5))
plt.plot([0,0.5],[0,0.5], '-')
plt.plot(y_hat, chl_situ[depth], '.', markersize = 6)
plt.xlim([0.10, 0.12])
plt.ylim([0.02, 0.2])
plt.xlabel("Prediction CHL-a conc."); plt.ylabel("Observed CHL-a conc."); plt.title("C2RCC (shallow)")
plt.show()
#%%
chl_situ = np.array(x_situ['chl'])

np.random.seed(101)
#chl_situ = chl_situ + np.random.normal(0, 0.005, size = len(chl_situ))

factor1, exponent1 = curve_fit(fun, xdata = iop_sat, ydata = chl_situ)[0]
factor2, exponent2 = curve_fit(fun, xdata = iop_sat[depth], ydata = chl_situ[depth])[0]

#factor, exponent = 20, 1

# Let's have a look of the data
plt.figure(figsize = (10,8))
plt.rcParams.update({'font.size': 18})
plt.scatter(iop_sat, chl_situ, color = 'b', label = 'depth >5 meter')
plt.scatter(iop_sat[depth], chl_situ[depth], color = 'red', label = 'depth <5 meter')
plt.plot(iop_sat[depth], factor2*iop_sat[depth]**exponent2, 'r', linewidth = 4)
plt.plot(iop_sat, factor1*iop_sat**exponent1, 'b', linewidth = 4)
plt.xlabel('IOP_apig')
plt.ylabel('Chlorophyll-a concentration')
plt.legend()
plt.show()

#what is the mean squared error?
y_hat = fun(iop_sat, factor1, exponent1)
err = y_hat - chl_situ
MSE = np.mean(err**2)
print(f"The MSE = {MSE}")

err = fun(iop_sat[depth], factor2, exponent2) - chl_situ[depth]
MSE = np.mean(err**2)
print(f"The MSE = {MSE}")

#%% Predict 

chl_situ = np.array(x_situ['chl'])
fac, exp = curve_fit(fun, xdata = iop_sat, ydata = chl_situ)[0]

iop[iop == 0] = np.nan

y_hat = fun(iop, fac, exp)
plotgrid(y_hat, lon, lat, coord, x_situ, 'Concentration Chlorophyll-a using C2RCC', plot_insitu=False, var_insitu = 'chl')
