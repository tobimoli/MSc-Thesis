#R# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:09:48 2021

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
import itertools
import GPy

#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
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

#Lo1, Lo2, La1, La2 = 24.80,25.40,40.20,40.65
Lo1, Lo2, La1, La2 = 25.14,25.16,40.31,40.33
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame([lon,lat,blue,green,red,NIR]).T
x_sat.columns = ['lon','lat','blue','green','red','NIR']

# Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
# NAME = 'ODYSSEA.csv'
# data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
# data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
# data['year'] = [int(n[6:10]) for n in data['time']]
# data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
# data['date'] = data['date'].astype("datetime64")

# data = data[data['date'] == '2019-08-02']

NAME = 'ODYSSEA_unique.csv'
data = pd.read_table(FOLDER+NAME, sep = ',')

data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#remove outliers
data = data[data['chl']>0]

x_situ = data[['lon', 'lat', 'dep', 'chl']]
X = pd.concat([x_sat, x_situ], ignore_index=True, sort=False)
print(x_sat.shape)
b.RGB(subset = [la1,la2,lo1,lo2])
#%%GPR for finding reflectances
def kernel_function(x, y, sigma_f, l):
    """Define squared exponential kernel function."""       
    kernel = sigma_f**2 * np.exp(- (1/2)* np.dot((x-y)**2, 1/l**2))
    return kernel

def compute_cov_matrices(x, x_star, sigma_f, l):
    """
    Compute components of the covariance matrix of the joint distribution.
    We follow the notation:
        - K = K(X, X) 
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    x = np.array(x); x_star = np.array(x_star)
    n = x.shape[0]
    n_star = x_star.shape[0]

    K = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x, x)]
    K = np.array(K).reshape(n, n)
    K_star2 = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x_star)]
    K_star2 = np.array(K_star2).reshape(n_star, n_star)
    K_star = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x)]
    K_star = np.array(K_star).reshape(n_star, n)
    
    return (K, K_star2, K_star)

def compute_gpr_parameters(K, K_star2, K_star, y, sigma_n):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    Y = np.array(y).reshape([n, d])
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), Y))
    # Covariance.
    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), K_star.T))
    
    return (f_bar_star, cov_f_star)

def plotmatrix(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data=matrix, cmap='Blues', ax=ax)
    ax.set(title=title)

def plotgrid(y, title = '', plot_insitu = False, var_insitu = 'chl'):
    matplotlib.rcParams['figure.figsize'] = (10,10)
    # Initialize map
    proj=ccrs.Mercator()
    m = plt.axes(projection=proj)
    plt.rcParams.update({'font.size': 18})
    # Plot data
    plt.pcolormesh(lon.reshape(la2-la1,lo1-lo2),
                   lat.reshape(la2-la1,lo1-lo2),
                   y.reshape(la2-la1,lo1-lo2), 
                   transform=ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))
    # Add Colorbar
    #plt.clim(0,0.3)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label('mg m-3', fontsize = 18)
    if plot_insitu:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = x_situ[var_insitu][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
                  edgecolor='black', linewidth=1.3, s = 150, vmin = np.min(y), vmax = np.max(y),
                  transform=ccrs.PlateCarree())
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
n = x_sat.shape[0]
n_star = x_situ.shape[0]
l = np.array([0.1, 0.1])
sigma_f = 2
sigma_n = 1
d = 4
#x_situ = x_situ[(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))]

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
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi
import time

start = time.time()
for i in range(1):
    #Q = linear_rbf(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
    P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)
start = time.time()
#GPR to find reflectances
# rbf = GPy.kern.RBF(input_dim=2, variance=sigma_f, lengthscale=l, ARD = True)
# gpr = GPy.models.GPRegression(np.array(x_sat[['lon', 'lat']]), np.array(x_sat[['blue','green','red','NIR']]), rbf)
# gpr.Gaussian_noise.variance = sigma_n**2

# gpr.optimize()
end = time.time()
print(end - start)
start = time.time()

# R = gpr.predict(np.array(x_situ[['lon','lat']]), full_cov= False, include_likelihood= False)
end = time.time()
print(end - start)
# K, K_star2, K_star = compute_cov_matrices(x_sat[['lon', 'lat']], x_situ[['lon','lat']], 
#                                           sigma_f=sigma_f, l=l)
# f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, x_sat[['blue','green','red','NIR']], sigma_n)

#add reflectances and depth to the datasets
x_situ[['blue', 'green', 'red', 'NIR']] = P
x_sat['dep'] = 1*np.ones(len(x_sat))

#%% GPR for finding chl-a

X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.array(x_situ[['chl']])
sigma_n = 1
sigma_f = 1
l = np.array([1, 1, 100, .1, .1, 1, 1])
d = 1

start = time.time()
for i in range(1):
    rbf = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
    gpr = GPy.models.GPRegression(X_train, Y_train, rbf)
    
    #print(gpr)
    
    # Fix the noise variance to known value 
    gpr.Gaussian_noise.variance = sigma_n**2
    
    # Run optimization
    gpr.optimize();

#print(gpr)
end = time.time()
print(end - start)

# Obtain optimized kernel parameters
l_opt = gpr.rbf.lengthscale.values
sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])
sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
np.set_printoptions(suppress=True)
#print(np.around(l_opt,3))
np.set_printoptions(suppress=False)

# Include noise into predictions using Equation (9).
#cov_s = cov_s + noise**2 * np.eye(*cov_s.shape)

#%% Predict using build-in function

start = time.time()
for i in range(1):
    prediction = gpr.predict(X, full_cov= False, include_likelihood= False)
end = time.time()
print(end - start)

plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Predict using own functions
K, K_star2, K_star = compute_cov_matrices(X_train, X, sigma_f=sigma_f_opt, l=l_opt)

f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, Y_train, sigma_n = sigma_n_opt)
#plotmatrix(cov_f_star, 'Components of the Covariance Matrix cov_f_star')

plotgrid(f_bar_star, 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(np.diag(cov_f_star)), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Plot reflectances
plotgrid(np.array(x_sat['blue']), title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue')
plotgrid(np.array(x_sat['green']), title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = True, var_insitu = 'green')
plotgrid(np.array(x_sat['red']), title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = True, var_insitu = 'red')
plotgrid(np.array(x_sat['NIR']), title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = True, var_insitu = 'NIR')