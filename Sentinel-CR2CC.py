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
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from netCDF4 import Dataset
import time

#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled_C2RCC"
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
#Lo1, Lo2, La1, La2 = 25, 25.40,39.75,40.05 #limnos

lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

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
    plt.clim(0.09,0.14)
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
def IDW(x1, y1, z):
    zi = np.zeros((len(x1), z.shape[1]))
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
    plt.errorbar(xval[:n], Prediction[0,:n], yerr=2*Prediction[1,:n].flatten(), fmt='.k', label = 'Predicted value');
    plt.plot(xval[:n], Y_train[:n], 'r.', markersize = 8, label = 'Observed value')
    plt.grid(True)
    #plt.ylim([0.0,0.9])
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
plt.plot([0,0.5],[0,0.5], '-'); plt.plot(y_hat, chl_situ, '.', markersize = 6); 
plt.xlabel("Prediction CHL-a conc."); plt.ylabel("Observed CHL-a conc."); plt.title("C2RCC (all)")
plt.show()
#%% Non-linear least squares DATA ONLY DEPTH <5 meters
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
chl_situ = np.array(x_situ['chl'])
def fun(x, a, b):
    y = a*x**b
    return y

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
plt.plot([0,0.5],[0,0.5], '-'); 
plt.plot(y_hat, chl_situ[depth], '.', markersize = 6);
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
plotgrid(y_hat, 'Concentration Chlorophyll-a using C2RCC', plot_insitu=False, var_insitu = 'chl')
