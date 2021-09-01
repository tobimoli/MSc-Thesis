# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:31:58 2021

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
#name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"
#name = "subset_of_S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"
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
#Lo1, Lo2, La1, La2 = 24.8,25.20,40.40,40.70 #cloud
#Lo1, Lo2, La1, La2 = 24.0,26.20,38.40,42.70 #full map
Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map
#Lo1, Lo2, La1, La2 = 25.1,25.15,40.32,40.36 #map xs

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

# Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA.csv'
#NAME = 'ODYSSEA_unique.csv'

#data = pd.read_table(FOLDER+NAME, sep = ',')


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
    plt.clim(0,3)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label('mg m-3', fontsize = 18)
    if plot_insitu:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = x_situ[var_insitu][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
                  edgecolor='black', linewidth=1.3, s = 150, vmin = np.nanmin(y), vmax = np.nanmax(y),
                  transform=ccrs.PlateCarree(), cmap = plt.get_cmap(cmap))
    else:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = 'r', transform=ccrs.PlateCarree())
    #m.set_extent([Lo1,Lo2,La1,La2])
    m.coastlines(resolution="10m")
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
def IDW(x1, y1):
    zi = np.zeros((len(x1), 4))
    for i in range(len(x1)):
        close = np.array(find_closest(x1[i], y1[i], 10))
        dist = np.sqrt((close[:, 0] - x1[i] )**2 + (close[:, 1] - y1[i])**2)
        weights = 1.0 / dist**2
        weights /= weights.sum(axis = 0)
        zi[i] = np.dot(weights.T, close[:, 2:])
    return zi
def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist**2

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
def plot_errbar(Y_train, Prediction, xval, xlabel, n = 1000):
    plt.figure(figsize = (10, 8))
    plt.rcParams.update({'font.size': 18})
    plt.errorbar(xval[:n], Prediction[0,:n], yerr=2*np.sqrt(Prediction[1,:n]).flatten(), fmt='.k', label = 'Predicted value');
    plt.plot(xval[:n], Y_train[:n], 'r.', markersize = 8, label = 'True value')
    plt.grid(True)
    plt.ylim([0.0,0.5])
    plt.xlabel(xlabel)
    plt.ylabel('Chl-a concentration')
    plt.legend()
    plt.show()
def plot_errcont(xval, Y_train, Prediction, xfit, xlabel):
    plt.figure(figsize = (10, 8))
    plt.rcParams.update({'font.size': 18})
    #plt.plot(xval, Y_train, 'r.', markersize = 8, label = 'True value')
    yfit = Prediction[0].flatten()
    dyfit = 2 * np.sqrt(Prediction[1].flatten())
    
    plt.plot(xfit, yfit, '-', color='gray', label = 'Posterior Mean')
    plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2, label = "2 std range")
    
    plt.grid(True)
    plt.ylim([0.0,0.5])
    plt.xlabel(xlabel)
    plt.ylabel('Chl-a concentration')
    plt.legend()
    plt.show()

x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]

start = time.time()
for i in range(1):
    #P = IDW(np.array(x_situ['lon']), np.array(x_situ['lat']))
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
gpr.optimize(max_iters = 20, messages = True); 
 
#l_opt = gpr.rbf.lengthscale.values
#sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])   
#sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])

mean, var = gpr.predict(X, full_cov= False, include_likelihood= True)
prediction = np.exp(mean + 0.5*var), (np.exp(var)-1)*np.exp(2*mean+var)
e = time.time()
print(e-s)
plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

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
    
    plotgrid(Prediction[0], 'Concentration Chlorophyll-a using SGPR', plot_insitu=True, var_insitu = 'chl')
    plotgrid(np.sqrt(Prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using SGPR')
    
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