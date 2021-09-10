# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:08:47 2021

@author: molenaar
"""
#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
import time

from Function_file import Data, simple_idw, plot_errcont, plotgrid
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
#Lo1, Lo2, La1, La2 = 25.1,25.15,40.32,40.36 #map xs

lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)
coord = [lo1, lo2, la1, la2]

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()
Class = b.var_values['quality_scene_classification'][la1:la2,lo1:lo2].flatten()

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
#%% ONLY run this when you want to select random rows to reduce size
import random
random.seed(10)
randomlist = random.sample(range( len(x_situ)), 350)

x_situ = x_situ.loc[x_situ.index[randomlist],:]
#%%GPR for finding reflectances
x_sat.loc[Class != 6, :] = np.nan

start = time.time()
P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

#plotgrid(np.array(x_sat['blue']), plot_insitu = True)
#%% GPR for finding chl-a
x_sat.loc[x_sat['NIR']>0.1,:] = np.nan

X = np.array(x_sat[['green', 'red']])
X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['green', 'red']])
X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])

Y_train = np.log(np.array(x_situ[['chl']]))
Prediction = np.zeros((2, len(Y_train)))

l = np.array([99.963, 99.999, 75.506,  0.009,  0.396,  0.009,  0.009])
sigma_n = 0.223
sigma_f = 0.668

N1 = 20; X1 = np.logspace(-4, 0, N1)
N2 = 20; X2 = np.logspace(-1, 3, N2)
ll = np.zeros(N1*N2)
i = 0
for L1 in X1:
    print(i)
    l[6] = L1
     
    for L2 in X2:
        l[0] = L2
        kern = GPy.kern.sde_Matern32(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
        gpr = GPy.models.GPRegression(X_train, Y_train, kern, normalizer=True)  
        gpr.Gaussian_noise.variance = sigma_n**2
        ll[i] = gpr.log_likelihood()
        
        i+=1

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize = (10,8))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('lengthscale for NIR wavelength')
ax.set_ylabel('lengthscale for longitude')

CS = ax.contourf(X1, X2, ll.reshape(N1,N2).T, 20, cmap=plt.cm.bone)
CS2 = ax.contour(CS, levels=CS.levels[::2], colors='r')
#ax.clabel(CS2, inline=True, fontsize=18)
cbar = fig.colorbar(CS)
cbar.add_lines(CS2)
cbar.ax.set_ylabel('Log Marginal Likelihood')
#%% 1D ll versus parameter
sigma_n = 0.215
sigma_f = 0.795
l = np.array([.24, 9.5, 27.5, 13.5, 0.022, 5.42, .0168])
txt = ['longitude', 'latitude', 'depth', 'blue wavelength', 'green wavelength', 'red wavelength', 'NIR wavelength', 'sigma_f', 'sigma_n']

X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.log(np.array(x_situ[['chl']]))

N2 = 100
X2 = np.logspace(-3, 3, N2)
ll = np.zeros((N2,len(l)+2))

for j in range(len(l)+2):
    print(txt[j])
    i = 0
    #l = np.array([1.,1.,1.,1.,1.,1.,1.])
    l = np.array([99.963, 99.999, 75.506,  0.009,  0.396,  0.009,  0.009])
    #l = np.array([0.15, 10.1, 78.5, 41.7, 53.8, 4.29, 0.106])
    sigma_n = 0.223
    sigma_f = 0.668
    for L2 in X2:
        print(i)
        if j == 7:
            sigma_f = L2
        elif j == 8:
            sigma_n = L2
        else:
            l[j] = L2
        rbf = GPy.kern.sde_Matern32(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
        gpr = GPy.models.GPRegression(X_train, Y_train, rbf, normalizer=True)        
        gpr.Gaussian_noise.variance = sigma_n**2
        ll[i,j] = gpr.log_likelihood()
        
        i+=1
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize = (10,8))
    ax.set_xscale("log")
    if j ==7:
        ax.set_xlabel('process variance ($\sigma_f^2$)')
    elif j ==8:
        ax.set_xlabel('noise std. ($\sigma_n$)')
    else:
        ax.set_xlabel('lengthscale for '+ txt[j])
    ax.set_ylabel('Log Marginal Likelihood')
    ax.grid(True, which = 'both')
    ax.plot(X2, ll[:,j])
    plt.show()
#%% Predict using build-in function

l = np.array([99.963, 99.999, 75.506,  0.009,  0.396,  0.009,  0.009])
sigma_n = 0.223
sigma_f = 0.668

rbf = GPy.kern.sde_Matern32(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
gpr = GPy.models.GPRegression(X_train, Y_train, rbf)
gpr.Gaussian_noise.variance = sigma_n**2


N = 500
row = 410
X_cont = np.outer(np.ones(N),X_train[row])
X_cont[:,2] = np.linspace(0, 500, N)

mean, var = gpr.predict(X_cont, full_cov= False, include_likelihood= True)
prediction = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
plot_errcont(X_train[X_train[:,0] == X_train[row,0],2], np.exp(Y_train[X_train[:,0] == X_train[row,0]]), prediction, X_cont[:,2], 'Depth [m]')
#plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
#plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

N = 500
row = 410
X_cont = np.outer(np.ones(N),X_train[row])
X_cont[:,0] = np.linspace(10, 50, N)

mean, var = gpr.predict(X_cont, full_cov= False, include_likelihood= True)
prediction = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
plot_errcont(X_train[X_train[:,0] == X_train[row,0],2], np.exp(Y_train[X_train[:,0] == X_train[row,0]]), prediction, X_cont[:,0], 'longitude')


N = 500
row = 410
X_cont = np.outer(np.ones(N),X_train[row])
X_cont[:,3] = np.linspace(0.02, 0.05, N)

mean, var = gpr.predict(X_cont, full_cov= False, include_likelihood= True)
prediction = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
plot_errcont(X_train[X_train[:,0] == X_train[row,0],2], np.exp(Y_train[X_train[:,0] == X_train[row,]]), prediction, X_cont[:,3], 'blue wavelength')

#%% Plot reflectances
x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]

plotgrid(np.array(x_sat['blue']), lon, lat, coord, x_situ, title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue', cmap = 'Blues')
plotgrid(np.array(x_sat['green']), lon, lat, coord, x_situ, title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = False, var_insitu = 'green', cmap = 'Greens')
plotgrid(np.array(x_sat['red']), lon, lat, coord, x_situ, title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = True, var_insitu = 'red', cmap = 'Reds')
plotgrid(np.array(x_sat['NIR']), lon, lat, coord, x_situ, title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = False, var_insitu = 'NIR', cmap = 'RdPu')

#%%

for key in b.keys:
    plt.figure()
    plt.imshow(b.var_values[key])
    plt.title(b.var_names[key])
    plt.show()
    
