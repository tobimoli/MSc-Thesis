# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:22:33 2021

@author: molenaar
"""
#%% load packages

import numpy as np
import pandas as pd
import GPy
import time
from Function_file import Data, simple_idw, plot_errbar, plot_errcont, plotgrid
#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
#name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"
#name = "subset_of_S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"

b = Data(folder, name)
b.import_data()

#%% select area
#Lo1, Lo2, La1, La2 = 24.8,25.20,40.40,40.70 #cloud
#Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #full map
Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map

lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)
coord = [lo1, lo2, la1, la2]

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame(np.array([lon,lat,blue,green,red,NIR]).T)
x_sat.columns = ['lon','lat','blue','green','red','NIR']

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
#b.RGB(subset = [la1,la2,lo1,lo2])
#%%GPR for finding reflectances
start = time.time()
for i in range(1):
    P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

#plotgrid(np.array(x_sat['blue']), plot_insitu = True)
#%% GPR for finding chl-a
x_sat.loc[x_sat['NIR']>0.1,:] = np.nan

X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.array(x_situ[['chl']])
Prediction = np.zeros((2, len(Y_train)))

sigma_n = 1
sigma_f = 1
l = np.array([10, 10, 100, .1, .1, 1, 1])
from sklearn.model_selection import KFold
K = 10; i = 0
kf = KFold(n_splits=K, shuffle = True, random_state = 0)
mse = np.zeros(K)
for train, test in kf.split(X_train):
    
    start = time.time()
    
    rbf = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
    gpr = GPy.models.GPRegression(X_train[train], Y_train[train], rbf)
    
    #print(gpr)
    end = time.time()
    #print(end - start)
    start = time.time()
    
    # Fix the noise variance to known value 
    gpr.Gaussian_noise.variance = sigma_n**2
    
    # Run optimization
    gpr.optimize(max_iters = 10, messages = True);

    #print(gpr)
    end = time.time()
    #print(end - start)

    # Obtain optimized kernel parameters
    l_opt = gpr.rbf.lengthscale.values
    sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])
    sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
    print(sigma_n_opt**2)
    np.set_printoptions(suppress=True)
    #print(np.around(l_opt,3))
    np.set_printoptions(suppress=False)
    
    start = time.time()
    mean, var = gpr.predict(X_train[test], full_cov= False, include_likelihood= True)
    end = time.time()
    #print(end - start)
    
    Prediction[0, test], Prediction[1, test] = mean.T, var.T
    mse[i] = np.mean((Y_train[test]-Prediction[0, test])**2)
    i+=1

xval = X_train[:, 2]
xval = range(len(Y_train))
plot_errbar(Y_train, Prediction, xval, 'Depth', n = 500)

tot_in_err = np.sum((Y_train.flatten() < Prediction[0] + 2*np.sqrt(Prediction[1]).flatten()) & (Y_train.flatten() > Prediction[0] - 2*np.sqrt(Prediction[1]).flatten()))
print(tot_in_err)
print(tot_in_err/len(Y_train))
print("The MSE = " + str(np.mean(mse)))

#%% Predict using build-in function

N = 500
row = 410
X_cont = np.outer(np.ones(N),X_train[row])
X_cont[:,2] = np.linspace(0, 500, N)

rbf = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
gpr = GPy.models.GPRegression(X_train, Y_train, rbf)
gpr.Gaussian_noise.variance = sigma_n**2
gpr.optimize(max_iters = 20, messages = True); 
    
prediction = gpr.predict(X_cont, full_cov= False, include_likelihood= True)

plot_errcont(X_train[X_train[:,0] == X_train[row,0],2], Y_train[X_train[:,0] == X_train[row,0]], prediction, X_cont[:,2], 'depth')
#plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
#plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Plot reflectances
x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]

plotgrid(np.array(x_sat['blue']), lon, lat, coord, x_situ, title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue', cmap = 'Blues')
plotgrid(np.array(x_sat['green']), lon, lat, coord, x_situ, title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = False, var_insitu = 'green', cmap = 'Greens')
plotgrid(np.array(x_sat['red']), lon, lat, coord, x_situ, title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = True, var_insitu = 'red', cmap = 'Reds')
plotgrid(np.array(x_sat['NIR']), lon, lat, coord, x_situ, title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = False, var_insitu = 'NIR', cmap = 'RdPu')