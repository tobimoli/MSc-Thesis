# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:31:58 2021

@author: molenaar
"""

#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
import time

from Function_file import Data, simple_idw, plotgrid
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
coord = [lo1,lo2,la1,la2]

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
#%%GPR for finding reflectances
x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]

start = time.time()
P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

#%% GPR for finding chl-a
x_sat.loc[x_sat['NIR']>0.1,:] = np.nan
x_sat.loc[Class != 6, :] = np.nan

X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
#X = np.array(x_sat[['dep', 'blue', 'green', 'red', 'NIR']])
#X_train = np.array(x_situ[['dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.log(np.array(x_situ[['chl']]))

sigma_n = 0.1
sigma_f = 0.1
l = np.array([10., 10., 100., .1, .1, .1, .1])
#l = np.array([100., .1, .1, .1, .1])

#%% Predict using build-in function
np.random.seed(101)
M_range = [3, 30, 50, 100, 150, 200, 400, 800]
results = np.zeros((3,len(M_range)+1))

# X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
# Y_train = np.log(np.array(x_situ[['chl']]))
# random_indices = np.random.choice(X_train.shape[0], size=1001, replace=False)
# X_train = X_train[random_indices]
# Y_train = Y_train[random_indices]

kern = GPy.kern.sde_Matern32(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)

s = time.time()
gpr = GPy.models.GPRegression(X_train, Y_train, kern)
gpr.Gaussian_noise.variance = sigma_n**2
gpr.optimize(max_iters = 20, messages = True)

#l_opt = gpr.rbf.lengthscale.values
#sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])
#sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])

mean, var = gpr.predict(X, full_cov= False, include_likelihood= True)
prediction = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
e = time.time()
print(e-s)
plotgrid(prediction[0], lon, lat, coord, x_situ, 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), lon, lat, coord, x_situ, 'Std. Dev. of Concentration Chlorophyll-a using GPR')

results[:,-1] = X_train.shape[0], gpr.log_likelihood(), e - s

#%%
j = 0
for M in M_range[j:]:
    Z = np.zeros((M,7))
    for i in range(7):
        Z[:,i] = np.random.rand(M)*(max(X_train[:,i]) - min(X_train[:,i])) + min(X_train[:,i])

    start = time.time()
    m = GPy.models.SparseGPRegression(X_train,Y_train,kernel=kern,Z=Z)
    m.noise_var = 1
    m.optimize(max_iters = 100, messages=True)

    mean, var = m.predict(X, full_cov = False, include_likelihood = True)
    Prediction = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
    end = time.time()
    print(end-start)

    plotgrid(Prediction[0], lon, lat, coord, x_situ, 'Concentration Chlorophyll-a using SGPR', plot_insitu=True, var_insitu = 'chl')
    plotgrid(np.sqrt(Prediction[1]), lon, lat, coord, x_situ, 'Std. Dev. of Concentration Chlorophyll-a using SGPR')
    
    results[:,j] = M, m.log_likelihood()[0][0], end - start
    
    j += 1

#%% make a plot with different y-axis using second axis object
x = list(range(len(M_range)+1))
N = X_train.shape[0]

fig,ax = plt.subplots(figsize=(8,6))
plt.rcParams.update({'font.size': 18})
ax.plot(x, results[1], color="red", marker="o")
ax.set_xlabel("Number of inducing points m")
ax.set_ylabel("Log Likelihood",color="red")
ax2=ax.twinx()
ax2.bar(x, results[2] ,color="blue", align = 'center', alpha = 0.3)
ax2.set_ylabel("Time [s]",color="blue")
plt.title(f"n = {N}")
plt.xticks(x, results[0].astype(int))
plt.show()