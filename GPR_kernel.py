# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:47:42 2021

@author: molenaar
"""

#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
import time
from sklearn.model_selection import KFold
from Function_file import Data, simple_idw, plotgrid, plot_errcont
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
#%%IDW for finding reflectances
x_sat.loc[Class != 6, :] = np.nan

def choose_kernel(k, sigma_f, l = []):
    if k == 0:
        kern = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
    elif k == 1:
        kern = GPy.kern.sde_Matern52(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
    elif k==2:
        kern = GPy.kern.sde_Matern32(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
    elif k==3:
        kern = GPy.kern.OU(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
    elif k==4:
        kern = GPy.kern.MLP(input_dim=7, variance=sigma_f, weight_variance=l[0], bias_variance=1, ARD=True)
    elif k==5:
        kern = GPy.kern.Linear(input_dim=7, variances=sigma_f, ARD=False)
    return kern
def obtain_parameters(k):
    l_opt = []
    if k == 0:
        l_opt = gpr.rbf.lengthscale.values
        sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])
    elif k == 1:
        l_opt = gpr.Mat52.lengthscale.values
        sigma_f_opt = np.sqrt(gpr.Mat52.variance.values[0])
    elif k == 2:
        l_opt = gpr.Mat32.lengthscale.values
        sigma_f_opt = np.sqrt(gpr.Mat32.variance.values[0])
    elif k == 3:
        l_opt = gpr.OU.lengthscale.values
        sigma_f_opt = np.sqrt(gpr.OU.variance.values[0])
    elif k == 4:
        sigma_f_opt = np.sqrt(gpr.mlp.variance.values[0])
        l_opt = [np.sqrt(gpr.mlp.weight_variance.values[0])]
    elif k == 5:
        sigma_f_opt = np.sqrt(gpr.linear.variances.values[0])
    sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
    return sigma_n_opt, sigma_f_opt, l_opt

start = time.time()
for i in range(1):
    P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

#add clusters
j=0
cl = np.zeros(len(x_situ))
for index,row in x_situ.iterrows():
    if row['lon']<25.07:
        cl[j] = 1
    elif row['lon']<25.105:
        cl[j] = 2
    elif row['lon']<25.13:
        cl[j] = 3
    elif row['lon']<25.16:
        cl[j] = 4
    elif row['lon']<25.195:
        cl[j] = 5
    elif row['lat']<40.35:
        cl[j] = 6
    else:
        cl[j] = 7
    j+=1
x_situ['cluster'] = cl
#%% GPR for finding chl-a
#x_sat.loc[x_sat['NIR']>0.1,:] = np.nan
#x_situ = x_situ[x_situ['dep']<5]

X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.log(np.array(x_situ[['chl']]))
Prediction = np.zeros((2, len(Y_train)))

sigma_n = 1
sigma_f = 1
l = np.array([10, 10, 100, 1, 1, 1, 1])
K = 7; i = 0
kf = KFold(n_splits=K, shuffle = True, random_state = 0)
mse = np.zeros(K)
LL = np.zeros(K)

kernels = ['Squared Exponential', 'Matérn 5/2', 'Matérn 3/2', 'Matérn 1/2', 'MLP', 'Linear']
k = 2
for k in range(2,3):
    Y_train = np.log(np.array(x_situ[['chl']]))
    Prediction = np.zeros((2, len(Y_train)))

    kern = choose_kernel(k, sigma_f, l)
    #kern = GPy.kern.RatQuad(input_dim=7, variance=sigma_f, lengthscale=l, power = 2, ARD=False)
    #kern = GPy.kern.Poly(input_dim=7, variance=sigma_f, order=3, ARD=False)
    
    start = time.time()
    #for train, test in kf.split(X_train):
    for i in range(K):
        train, test = (np.array(x_situ['cluster']) != (i+1), np.array(x_situ['cluster']) == (i+1))
        print(i)
        gpr = GPy.models.GPRegression(X_train[train], Y_train[train], kern, normalizer = True)
        
        # Fix the noise variance to known value 
        gpr.Gaussian_noise.variance = sigma_n**2
        
        # Run optimization
        gpr.optimize(max_iters = 100, messages = True)
    
        # Compute the MSE using the test set
        mean, var = gpr.predict(X_train[test], full_cov= False, include_likelihood= True)
        mean, var = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
        
        Prediction[0, test], Prediction[1, test] = mean.T, var.T
        mse[i] = np.mean((np.exp(Y_train[test].T)-Prediction[0, test])**2)
        
        # Compute the LL using the test set
        sigma_n_opt, sigma_f_opt, l_opt = obtain_parameters(k)
        kern = choose_kernel(k, sigma_f_opt, l_opt)
        gpr = GPy.models.GPRegression(X_train[test], Y_train[test], kern, normalizer = True)
        gpr.Gaussian_noise.variance = sigma_n_opt**2
        
        LL[i] = gpr.log_likelihood()
    
        i+=1
    
    end = time.time()
    print("Time per iteration = " + str(np.round((end - start)/K,3)) + "s")
    
    Y_train = np.exp(Y_train)
    
    xval = X_train[:, 2]
    xval = range(len(Y_train))
    #plot_errbar(Y_train, Prediction, xval, 'Time', n = 500)
    tot_in_err = np.sum((Y_train.flatten() < Prediction[0] + 2*np.sqrt(Prediction[1]).flatten()) & (Y_train.flatten() > Prediction[0] - 2*np.sqrt(Prediction[1]).flatten()))
    print(kernels[k])
    print("Obs. in range = " + str(tot_in_err) + "/" + str(len(Y_train)))
    print("Perc. in range: " + str(np.round(tot_in_err/len(Y_train),3)) + "%")
    print("The MSE (s.e.) = " + str(np.round(np.mean(mse),5)) + ' (' + str(np.round(mse.std()/np.sqrt(K),5)) + ')')
    print("Mean Pred.std = " + str(np.round(np.mean(Prediction[1,:]),5)))
    print("Marg. Log. Likelihood = " + str(np.round(np.mean(LL),3)) + ' (' + str(np.round(LL.std()/np.sqrt(K),3)) + ')')
    print("R-squared = " + str(1 - np.sum((Prediction[0] - Y_train.T)**2)/np.sum((Y_train - np.mean(Y_train))**2)))
    print("AIC = " + str(len(Y_train)*np.log(np.mean(mse)) + 2*(len(l_opt)+2)))
    print("BIC = " + str(len(Y_train)*np.log(np.mean(mse)) + np.log(len(Y_train))*(len(l_opt)+2)))
    plt.plot(Prediction[0], Y_train, '.', markersize = 2); plt.plot([0,.5],[0,.5], '-')
#%% Predict using build-in function

N = 500
row = 0
X_cont0 = np.outer(np.ones(N),X_train[row]); X_cont2 = np.outer(np.ones(N),X_train[row])
X_cont3 = np.outer(np.ones(N),X_train[row])
X_cont0[:,0] = np.linspace(25.0, 25.4, N)
X_cont2[:,2] = np.linspace(0,500,N)
X_cont3[:,3] = np.linspace(0.035, 0.040, N)

Y_train = np.log(np.array(x_situ[['chl']]))

sigma_n = .1
sigma_f = .1
l = np.array([10, 10, 100, 1, 1, 1, 1])
kernels = ['Squared Exponential', 'Matérn 5/2', 'Matérn 3/2', 'Matérn 1/2', 'MLP', 'Linear']
k = 2

if k == 0:
    kern = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
elif k == 1:
    kern = GPy.kern.sde_Matern52(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
elif k==2:
    kern = GPy.kern.sde_Matern32(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
elif k==3:
    kern = GPy.kern.OU(input_dim=7, variance=sigma_f, lengthscale=l, ARD=True)
elif k==4:
    kern = GPy.kern.MLP(input_dim=7, variance=sigma_f, weight_variance=l, bias_variance=1, ARD=True)
else:
    kern = GPy.kern.Linear(input_dim=7, variances=sigma_f, ARD=False)

gpr = GPy.models.GPRegression(X_train, Y_train, kern, normalizer = True)
gpr.Gaussian_noise.variance = sigma_n**2
gpr.optimize(max_iters = 200, messages = True, optimizer = 'lbfgs')

l_opt = gpr.Mat32.lengthscale.values
sigma_f_opt = np.sqrt(gpr.Mat32.variance.values[0])
sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
    
prediction = gpr.predict(X, full_cov= False, include_likelihood= True)
mu, sigma = np.exp(prediction[0] + 0.5*prediction[1]), (np.exp(prediction[1])-1)*np.exp(2*prediction[0]+prediction[1])

plotgrid(mu, lon, lat, coord, x_situ, title = 'Concentration CHL-a with a '+kernels[k]+' kernel', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(sigma), lon, lat, coord, x_situ, title ='SD of Concentration CHL-a with a '+kernels[k]+' kernel')
print(mu.std())

prediction = gpr.predict(X_cont0, full_cov= False, include_likelihood= True)
mu, sigma = np.exp(prediction[0] + 0.5*prediction[1]), (np.exp(prediction[1])-1)*np.exp(2*prediction[0]+prediction[1])
plot_errcont('a','b', [mu, sigma], X_cont0[:,0], 'Longitude [deg]', kernels[k])

prediction = gpr.predict(X_cont2, full_cov= False, include_likelihood= True)
mu, sigma = np.exp(prediction[0] + 0.5*prediction[1]), (np.exp(prediction[1])-1)*np.exp(2*prediction[0]+prediction[1])
plot_errcont('a','b', [mu, sigma], X_cont2[:,2], 'Depth [m]', kernels[k])

prediction = gpr.predict(X_cont3, full_cov= False, include_likelihood= True)
mu, sigma = np.exp(prediction[0] + 0.5*prediction[1]), (np.exp(prediction[1])-1)*np.exp(2*prediction[0]+prediction[1])
plot_errcont('a','b', [mu, sigma], X_cont3[:,3], 'Reflectence Blue Wavelength', kernels[k])

#%% Plot reflectances
x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]
x_situ = x_situ[(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))]

plotgrid(np.array(x_sat['blue']), lon, lat, coord, x_situ, title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue', cmap = 'Blues')
plotgrid(np.array(x_sat['green']), lon, lat, coord, x_situ, title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = False, var_insitu = 'green', cmap = 'Greens')
plotgrid(np.array(x_sat['red']), lon, lat, coord, x_situ, title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = True, var_insitu = 'red', cmap = 'Reds')
plotgrid(np.array(x_sat['NIR']), lon, lat, coord, x_situ, title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = False, var_insitu = 'NIR', cmap = 'RdPu')

#%%
plt.figure(figsize = (8,8))
plt.plot([0,0.5],[0,0.5], '-')
plt.plot(Prediction[0], Y_train, '.', markersize = 6)

plt.xlabel("Prediction CHL-a conc."); plt.ylabel("Observed CHL-a conc."); plt.title("Matérn")
plt.show()