# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:22:33 2021

@author: molenaar
"""
#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from netCDF4 import Dataset
import GPy
import time

#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"

class Data:
    def __init__(self):
        self.var = ''
        self.keys = []
        self.var_values = {}
        self.var_units = {}
        self.var_names = {}
        self.df = pd.DataFrame()
        self.mask = []
    def import_data(self):
        file = folder + "/" + name + ".nc"
        dataset = Dataset(file, "r")
        self.keys = list(dataset.variables.keys())
        for key in self.keys:
            self.var_values[key] = dataset.variables[key][:]
            try:
                self.var_names[key] = dataset.variables[key].long_name
            except:
                self.var_names[key] = key
            try:
                self.var_units[key] = dataset.variables[key].units
            except:
                self.var_units[key] = '-'
        dataset.close()
        print('The data is imported')
    def hist(self, var, ymax = 0):
        self.var = var
        plt.figure(1, figsize = (10,5))
        plt.grid()
        plt.hist(self.var_values[self.var].flatten(), bins=100, label = self.var, alpha = 0.5)
        plt.legend()
        if ymax == 0:
            plt.ylim()
        else:
            plt.ylim([0,ymax])
        plt.plot()   
    def boxplot(self):
        fig, ax = plt.subplots(1, 4, figsize = (15,5))
        i=0
        for key in self.keys:
            if key != 'lon' and key != 'lat':
                ax[i].boxplot(self.var_values[key].flatten())
                ax[i].set_title(key)
                i+=1
        plt.show()
    def plot(self, var, subset = [0,-1,0,-1]):
        self.var = var
        matplotlib.rcParams['figure.figsize'] = (10,10)
        # Initialize map
        proj=ccrs.Mercator()
        m = plt.axes(projection=proj)
        plt.rcParams.update({'font.size': 18})
        # Plot data
        plt.pcolormesh(self.var_values['lon'][subset[0]:subset[1],subset[2]:subset[3]],
                       self.var_values['lat'][subset[0]:subset[1],subset[2]:subset[3]],
                       self.var_values[self.var][subset[0]:subset[1],subset[2]:subset[3]], 
                       transform=ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))
        
        #shp = shapereader.Reader('C:/Users/molenaar/Downloads/gadm36_GRC_shp/gadm36_GRC_0')
        #for record, geometry in zip(shp.records(), shp.geometries()):
        #    m.add_geometries([geometry], ccrs.PlateCarree(), alpha = 0.5, facecolor='lightgray', edgecolor='black')

        gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2,
                               color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        gl.right_labels = False
        
        # Add Colorbar
        cbar = plt.colorbar(fraction=0.03, pad=0.04)
        cbar.set_label(self.var_units[self.var], fontsize = 18)
        
        # Add Title
        plt.title(self.var_names[self.var], fontsize = 18)
        plt.show()
    def dataframe(self):
        for key in self.keys:
            self.df[key] = self.var_values[key].flatten()[~self.mask]
    def correlation(self):
        plt.figure(figsize = (8,8))
        corrMatrix = self.df.corr()
        sns.heatmap(corrMatrix, annot=True)
        plt.show()
        sns.pairplot(self.df.sample(n = 10000, random_state = 44))
    def masktotal(self):
        mask = np.zeros(len(self.var_values['lon'].flatten()))
        for key in self.keys:
            mask += ma.getmask(self.var_values[key].flatten())
        self.mask = np.array(mask != 0)
    def RGB(self, subset = [0,-1,0,-1]):
        x,y = self.var_values['lon'][subset[0]:subset[1],subset[2]:subset[3]].shape
        retrack_original = np.zeros((x,y,3),dtype=int)
        MAX = [np.max(self.var_values['B4']), np.max(self.var_values['B3']), np.max(self.var_values['B2'])]
        for i in range(x):
            for j in range(y):
                retrack_original[i][j][0] = self.var_values['B4'].data[i+subset[0],j+subset[2]]/MAX[0]*255
                retrack_original[i][j][1] = self.var_values['B3'].data[i+subset[0],j+subset[2]]/MAX[1]*255
                retrack_original[i][j][2] = self.var_values['B2'].data[i+subset[0],j+subset[2]]/MAX[2]*255
        plt.imshow(retrack_original)
    def chl(self, alg):
        if alg == 0:
            self.var_values['chl'] = self.var_values['B2']/self.var_values['B3']
        elif alg == 1:
            self.var_values['chl'] = self.var_values['B8']/self.var_values['B4']
        self.var_units['chl'] = 'mg'
        self.var_names['chl'] = 'Chlorophyll-a Concentration ' + str(alg)
b = Data()
b.import_data()

#%% select area
#Lo1, Lo2, La1, La2 = 25, 25.40,39.75,40.05 #limnos
#Lo1, Lo2, La1, La2 = 25.6,26.00,40.05,40.30 #imbros
#Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #clouds
Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map
#Lo1, Lo2, La1, La2 = 24.5,25.30,39.80,40.50 #big map
#Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #all
#Lo1, Lo2, La1, La2 = 25.2,25.30,40.30,40.40 #xs

lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

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

#%%GPR for finding reflectances
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

x_sat.loc[Class != 6, :] = np.nan
x_sat.loc[NIR > .1, :] = np.nan

def plotmatrix(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data=matrix, cmap='Blues', ax=ax)
    ax.set(title=title)
def plotgrid(y, title = '', plot_insitu = False, var_insitu = 'chl', cmap = 'viridis'):
    matplotlib.rcParams['figure.figsize'] = (10,10)
    # Initialize map
    proj=ccrs.Mercator()
    m = plt.axes(projection=proj)
    plt.rcParams.update({'font.size': 18})
    # Plot data
    plt.pcolormesh(lon.reshape(la2-la1,lo1-lo2),
                   lat.reshape(la2-la1,lo1-lo2),
                   y.reshape(la2-la1,lo1-lo2), 
                   transform=ccrs.PlateCarree(), cmap = plt.get_cmap(cmap))
    # Add Colorbar
    #plt.clim(0.09,0.14)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label('mg m-3', fontsize = 18)
    if plot_insitu:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = x_situ[var_insitu][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
                  edgecolor='black', linewidth=1.3, s = 150, vmin = np.nanmin(y), vmax = np.nanmax(y),
                  cmap = plt.get_cmap(cmap), transform=ccrs.PlateCarree())
    else:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon)) & (x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon)) & (x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))], 
                  c = 'r', transform=ccrs.PlateCarree())
    #m.set_extent([Lo1,Lo2,La1,La2])
    #m.coastlines(resolution="10m")
    m.add_feature(cfeature.OCEAN)
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor=cfeature.COLORS['land'], zorder=0)
    m.add_feature(land_10m)
    gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2,
                           color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    
    # Add Title
    plt.title(title, fontsize = 18)
    plt.show()
def find_closest(lon, lat, nr):
    return x_sat.iloc[((x_sat['lon']-lon)**2 + (x_sat['lat']-lat)**2).argsort()[:nr]]
def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)
def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist**2

    # Make weights sum to one
    weights /= np.nansum(weights, axis=0)
    
    w1 = np.ma.array(weights, mask = np.isnan(weights))
    z1 = np.ma.array(z, mask = np.isnan(z))
    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.ma.dot(w1.T, z1)
    return zi
def IDW(x1, y1):
    zi = np.zeros((len(x1), 4))
    for i in range(len(x1)):
        print(i)
        close = np.array(find_closest(x1[i], y1[i], 4))
        dist = np.sqrt((close[:, 0] - x1[i] )**2 + (close[:, 1] - y1[i])**2)
        weights = 1.0 / dist**2
        weights /= weights.sum(axis = 0)
        zi[i] = np.dot(weights.T, close[:, 2:6])
    return zi
def idw(x,y,z,xi,yi):
    k = 100000
    zi = np.zeros((len(xi), 4))
    for i in range(len(xi)):
        if i%100 == 0:
            print(i)
        arr = np.sqrt((x - xi[i])**2 + (y - xi[i])**2)
        pos = np.argpartition(arr, k)[:k]
        dist = arr[pos]
        weights = 1.0 / dist**2
        weights /= weights.sum()
        
        zi[i] = np.dot(weights, z[pos])
    return zi

start = time.time()
P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P


#%% GPR for finding chl-a
X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat', 'dep', 'blue', 'green', 'red', 'NIR']])
X = np.array(x_sat[['dep', 'blue', 'red', 'NIR']])
X_train = np.array(x_situ[[ 'dep', 'blue', 'red', 'NIR']])
Y_train = np.log(np.array(x_situ[['chl']]))

sigma_n = .67
sigma_f = .22
l = np.array([99.96,100.00,75.51,0.009,0.40,0.0011,0.009])
l = np.array([75.51,0.009,0.0011,0.009])
#uncomment next line if normalizer is False
l = np.array([165.69, 0.02, 0.019, 0.0189]); sigma_f = .77; sigma_n = 0.20875
#l = np.array([.175123905, .287553144, 952.56055502, 100.08807912, 99.94616086, 100.01608946, 100.11588438])
#sigma_n = 0.22; sigma_f = 2


kern = GPy.kern.sde_Matern32(len(l), variance=sigma_f**2, lengthscale=l, ARD = True)
gpr = GPy.models.GPRegression(X_train, Y_train, kern, normalizer = False)  
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

plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Predict iteratively for large satellite dataset
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))
x_sat.loc[Class != 6, :] = np.nan
x_sat.loc[NIR > .1, :] = np.nan
X = np.array(x_sat[['dep', 'blue', 'red', 'NIR']])

prediction = np.empty(len(X)), np.empty(len(X)); prediction[0][:], prediction[1][:] = np.nan, np.nan
isnan = np.isnan(X).any(axis=1)
X_predict = X[~isnan, :]
M = np.empty(len(X_predict)); V = np.empty(len(X_predict))

#Sn_inv = np.linalg.cholesky(kern.K(X_train)+np.eye(len(X_train))*sigma_n**2)
#Sn_inv_x_y = np.dot(Sn_inv, Y_train)

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

plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')
#%%
plotgrid(X[:,1], 'blue wavelength')
plotgrid(X[:,2], 'red wavelength')
plotgrid(X[:,3], 'NIR wavelength')


#%% SVD
from scipy.sparse.linalg import eigsh
#from ..util.normalizer import Standardize
Y = Y_train
def compute_gpr_parameters(K, K_star2, K_star, sigma_n):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)),Y.reshape([n, 1])))
    # Covariance.
    var_f_star = K_star2 - (np.dot(K_star, np.linalg.inv(K+(sigma_n**2)*np.eye(n)))*K_star).sum(-1)
    
    return (f_bar_star.flatten(), var_f_star.flatten())
def compute_gpr_parameters_svd(K, K_star2, K_star, sigma_n, k):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    [S, U] = eigsh(K + (sigma_n**2)*np.eye(n), k = k)
    
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.dot(U * 1/S, U.T), Y.reshape([n, 1])))
    # Covariance.
    var_f_star = K_star2 - (np.dot(K_star, np.dot(U * 1/S, U.T))* K_star).sum(-1)
    
    return (f_bar_star.flatten(), var_f_star.flatten())

#%% Compute posterior mean and covariance. 
K, K_star2, K_star = kern.K(X_train), kern.Kdiag(X), kern.K(X,X_train)
s = time.time()
mean, var = compute_gpr_parameters(K, K_star2, K_star, sigma_n)
var += sigma_n**2
mean, var = np.exp(mean + 0.5*var).flatten(), ((np.exp(var)-1)*np.exp(2*mean+var)).flatten()
e = time.time()
print(e-s)
plotgrid(mean, 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(var), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Compute posterior mean and std. with SVD
for k in range(1,20):
    s = time.time()
    mean_svd, var_svd = compute_gpr_parameters_svd(K, K_star2, K_star, sigma_n, k=k)
    var_svd += sigma_n**2
    mean_svd, var_svd = np.exp(mean_svd + 0.5*var_svd).flatten(), ((np.exp(var_svd)-1)*np.exp(2*mean_svd+var_svd)).flatten()
    e = time.time()
    
    print(f'k = {k}: mean: {np.round(np.mean(np.abs(mean-mean_svd)), 4)} {np.round((mean-mean_svd).std()/np.sqrt(len(mean)),4)} and std: {np.round(np.mean(np.abs(np.sqrt(var)-np.sqrt(var_svd))),4)} {np.round((np.sqrt(var)-np.sqrt(var_svd)).std()/np.sqrt(len(var)),4)}, time = {np.round(e-s,3)} s')
    plotgrid(mean_svd, f'CHL-a using GPR and SVD: k={k}', plot_insitu=True, var_insitu = 'chl')
    plotgrid(np.sqrt(var_svd), f'Std. Dev. CHL-a using GPR and SVD: k={k}')
