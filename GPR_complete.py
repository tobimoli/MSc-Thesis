# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:22:33 2021

@author: tobias molenaar
"""
#%% load packages
import time
import sys
import numpy as np
import pandas as pd
import GPy
sys.path.insert(0, 'C:/Users/molenaar/OneDrive - Stichting Deltares/Documents/Thesis - Deltares/GitHub/MSc-Thesis')
from Function_file import Data, simple_idw, plotgrid
from Function_file import compute_gpr_parameters, compute_gpr_parameters_svd

#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"

b = Data(folder, name)
b.import_data()

#%% select area
#Lo1, Lo2, La1, La2 = 25, 25.40,39.75,40.05 #limnos
#Lo1, Lo2, La1, La2 = 25.6,26.00,40.05,40.30 #imbros
#Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #clouds
#Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map
#Lo1, Lo2, La1, La2 = 24.5,25.30,40.0,40.50 #big map
Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #all
#Lo1, Lo2, La1, La2 = 25.2,25.30,40.30,40.40 #xs
#Lo1, Lo2, La1, La2 = 25.25,25.30,40.35,40.40 #xxs

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
print(x_sat.shape)

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

#%%IDW for finding reflectances
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))
x_sat.loc[Class != 6, :] = np.nan
x_sat.loc[NIR > .1, :] = np.nan

#can give errors for large datasets, try to use a small dataset first
start = time.time()
P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),
               np.array(x_sat[['blue', 'green', 'red', 'NIR']]),
               np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P

#%% GPR for finding chl-a
#uncomment next 2 lines when full dataset is used
#X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
#X_train = np.array(x_situ[['lon', 'lat', 'dep', 'blue', 'green', 'red', 'NIR']])
X = np.array(x_sat[['dep', 'blue', 'red', 'NIR']])
X_train = np.array(x_situ[[ 'dep', 'blue', 'red', 'NIR']])
Y_train = np.log(np.array(x_situ[['chl']]))

#optimized parameters
sigma_n = .67
sigma_f = .22
l = np.array([99.96,100.00,75.51,0.009,0.40,0.0011,0.009])
l = np.array([75.51,0.009,0.0011,0.009])
#uncomment next line if normalizer is False
#l = np.array([165.69, 0.02, 0.019, 0.0189]); sigma_f = .77; sigma_n = 0.20875

kern = GPy.kern.sde_Matern32(len(l), variance=sigma_f**2, lengthscale=l, ARD = True)
gpr = GPy.models.GPRegression(X_train, Y_train, kern, normalizer = True)
gpr.Gaussian_noise.variance = sigma_n**2

#gpr.optimize(messages = True)

#%% Predict using build-in function
isnan = np.isnan(X).any(axis=1)
X_predict = X[~isnan, :]
prediction = np.empty(len(X)), np.empty(len(X)); prediction[0][:], prediction[1][:] = np.nan, np.nan

start = time.time()
mean, var = gpr.predict(X_predict, full_cov= False, include_likelihood= True)
prediction[0][~isnan], prediction[1][~isnan] = np.exp(mean + 0.5*var).flatten(), ((np.exp(var)-1)*np.exp(2*mean+var)).flatten()
end = time.time()
print(end - start)

plotgrid(prediction[0], lon, lat, coord, x_situ, 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), lon, lat, coord, x_situ, 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Predict iteratively for large satellite dataset
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))
x_sat.loc[Class != 6, :] = np.nan
x_sat.loc[NIR > .1, :] = np.nan
X = np.array(x_sat[['dep', 'blue', 'red', 'NIR']])

prediction = np.empty(len(X)), np.empty(len(X)); prediction[0][:], prediction[1][:] = np.nan, np.nan
isnan = np.isnan(X).any(axis=1)
X_predict = X[~isnan, :]
M = np.empty(len(X_predict)); V = np.empty(len(X_predict))

q = 10
start = time.time()
for i in range(len(X_predict)//q):
    if i % 1000 == 0:
        print(i/(len(X_predict)//q))
    mean, var = gpr.predict(X_predict[i*q:(i+1)*q,:], full_cov= False, include_likelihood= True)
    M[i*q:(i+1)*q], V[i*q:(i+1)*q] = np.exp(mean + 0.5*var).flatten(), ((np.exp(var)-1)*np.exp(2*mean+var)).flatten()
end = time.time()
print(end-start)
prediction[0][~isnan] = M; prediction[1][~isnan] = V

plotgrid(prediction[0], lon, lat, coord, x_situ,'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), lon, lat, coord, x_situ,'Std. Dev. of Concentration Chlorophyll-a using GPR')
#%% Plot interpolated reflectances

plotgrid(np.array(x_sat['blue']), lon, lat, coord, x_situ, title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue', cmap = 'Blues')
plotgrid(np.array(x_sat['green']), lon, lat, coord, x_situ, title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = True, var_insitu = 'green', cmap = 'Greens')
plotgrid(np.array(x_sat['red']), lon, lat, coord, x_situ, title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = True, var_insitu = 'red', cmap = 'Reds')
plotgrid(np.array(x_sat['NIR']), lon, lat, coord, x_situ, title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = True, var_insitu = 'NIR', cmap = 'RdPu')

#%% Compute posterior mean and covariance exact (without GPy)
K, K_star2, K_star = kern.K(X_train), kern.Kdiag(X), kern.K(X,X_train)
T = np.zeros(10)
for i in range(10):
    s = time.time()
    mean, var = compute_gpr_parameters(K, K_star2, K_star, sigma_n, Y_train)
    var += sigma_n**2
    mean, var = np.exp(mean + 0.5*var).flatten(), ((np.exp(var)-1)*np.exp(2*mean+var)).flatten()
    e = time.time()
    print(e-s)
    T[i] = e-s
plotgrid(mean, lon, lat, coord, x_situ,'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(var), lon, lat, coord, x_situ,'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Compute posterior mean and std. with SVD
T2 = np.zeros(10)

for i in range(10):
    k=50
    s = time.time()
    mean_svd, var_svd = compute_gpr_parameters_svd(K, K_star2, K_star, sigma_n, k=k, Y=Y_train)
    var_svd += sigma_n**2
    mean_svd, var_svd = np.exp(mean_svd + 0.5*var_svd).flatten(), ((np.exp(var_svd)-1)*np.exp(2*mean_svd+var_svd)).flatten()
    e = time.time()
    print(e-s)
    T2[i] = e-s
    #print(f'k = {k}: mean: {np.round(np.mean(np.abs(mean[~isnan]-mean_svd[~isnan])), 4)} {np.round((mean[~isnan]-mean_svd[~isnan]).std()/np.sqrt(len(mean[~isnan])),4)} and std: {np.round(np.mean(np.abs(np.sqrt(var[~isnan])-np.sqrt(var_svd[~isnan]))),4)} {np.round((np.sqrt(var[~isnan])-np.sqrt(var_svd[~isnan])).std()/np.sqrt(len(var[~isnan])),4)}, time = {np.round(e-s,3)} s')
    #plotgrid(mean_svd, lon, lat, coord, x_situ, f'CHL-a using GPR and SVD: k={k}', plot_insitu=True, var_insitu = 'chl')
    #plotgrid(np.sqrt(var_svd), lon, lat, coord, x_situ, f'Std. Dev. CHL-a using GPR and SVD: k={k}')

#%% standard deviations for time complexity and error plots

print(T.mean(), T.std()/np.sqrt(10))
print(T2.mean(), T2.std()/np.sqrt(10))
plotgrid(mean_svd, lon, lat, coord, x_situ,'Concentration Chlorophyll-a using SVD', plot_insitu=True, var_insitu = 'chl')
plotgrid(abs(mean-mean_svd), lon, lat, coord, x_situ,'Absolute Error', plot_insitu=True, var_insitu = 'chl')

