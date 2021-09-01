# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:37:40 2021

@author: molenaar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

#--------------------------------------------
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
class Data:
    def __init__(self, subset = [0,-1,0,-1]):
        self.var = ''
        self.keys = []
        self.var_values = {}
        self.var_units = {}
        self.var_names = {}
        self.sub = subset
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
    def plot(self, var):
        self.var = var
        matplotlib.rcParams['figure.figsize'] = (10,10)
        # Initialize map
        proj=ccrs.Mercator()
        m = plt.axes(projection=proj)
        plt.rcParams.update({'font.size': 18})
        # Plot data
        plt.pcolormesh(self.var_values['lon'][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]],
                       self.var_values['lat'][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]],
                       self.var_values[self.var][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]], 
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
    def RGB(self):
        x,y = self.var_values['lon'][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]].shape
        retrack_original = np.zeros((x,y,3),dtype=int)
        MAX = [np.max(self.var_values['B4']), np.max(self.var_values['B3']), np.max(self.var_values['B2'])]
        for i in range(x):
            for j in range(y):
                retrack_original[i][j][0] = self.var_values['B4'].data[i+self.sub[0],j+self.sub[2]]/MAX[0]*255
                retrack_original[i][j][1] = self.var_values['B3'].data[i+self.sub[0],j+self.sub[2]]/MAX[1]*255
                retrack_original[i][j][2] = self.var_values['B2'].data[i+self.sub[0],j+self.sub[2]]/MAX[2]*255
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

Lo1, Lo2, La1, La2 = 25.11,25.13,40.34,40.35
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()

X = np.array([lon,lat,blue,green,red,NIR]).T

# Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA.csv'
data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
data['year'] = [int(n[6:10]) for n in data['time']]
data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']

NAME = 'ODYSSEA_unique.csv'
data = pd.read_table(FOLDER+NAME, sep = ',')

data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#remove outliers
data = data[data['chl']>0]

x_star = np.array(data[['lon','lat']])
y = np.array(data['chl'])

fig, ax = plt.subplots()
# Plot function f. 
sns.scatterplot(x=x[:,1], y=x[:,0], color='red', label = 'f(x)', ax=ax)
# Plot function components.
ax.set_title(r'Scatterplot of longitude versus chl-a concentration')

#--------------------------------------------

nx_star = 10
ny_star = 10

x_star = np.array(np.meshgrid(np.linspace(start = 25.06, stop = 25.23, num = nx_star), 
                              np.linspace(start = 40.32, stop = 40.365, num = ny_star))).T.reshape(-1,2)

def kernel_function(x, y, sigma_f=1, l=[0.1,0.1]):
    """Define squared exponential kernel function."""
    L = np.diag([l])
    kernel = sigma_f**2 * np.exp(- (x-y).T * L * (x-y) / 2)
    return kernel
        
l = [0.1, 0.1]
sigma_f = 1

import itertools

def compute_cov_matrices(x, x_star, sigma_f=1, l=1):
    """
    Compute components of the covariance matrix of the joint distribution.
    
    We follow the notation:
    
        - K = K(X, X) 
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    n = x.shape[0]
    n_star = x_star.shape[0]

    K = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x, x)]

    K = np.array(K).reshape(n, n)
    
    K_star2 = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x_star)]

    K_star2 = np.array(K_star2).reshape(n_star, n_star)
    
    K_star = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x)]

    K_star = np.array(K_star).reshape(n_star, n)
    
    return (K, K_star2, K_star)

x = X[:,:2]
K, K_star2, K_star = compute_cov_matrices(x, x_star, sigma_f=sigma_f, l=l)

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K, cmap='Blues', ax=ax)
ax.set(title='Components of the Kernel Matrix K')

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_star2, cmap='Blues', ax=ax)
ax.set(title='Components of the Kernel Matrix K_star2');

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_star, cmap='Blues', ax=ax)
ax.set(title='Components of the Kernel Matrix K_star');

#--------------------------------------------
sigma_n = 1;
n = 72

a = np.concatenate((K + (sigma_n**2)*np.eye(n), K_star), axis=0)
b = np.concatenate((K_star.T, K_star2), axis=0)
C = np.concatenate((a, b), axis=1)
np.all(C.T == C)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=C, cmap='Blues', ax=ax)
ax.set(title='Components of the Covariance Matrix C');

#--------------------------------------------

fig, ax = plt.subplots()

for i in range(0, 100):
    # Sample from prior distribution. 
    z_star = np.random.multivariate_normal(mean=np.zeros(72), cov=K_star2)
    # Plot function.
    sns.lineplot(x=x_star[:,0], y=z_star, color='blue', alpha=0.2, ax=ax)
    
# Plot "true" linear fit.
sns.scatterplot(x=x[:,0], y=X[:,2], color='red', label = 'f(x)', ax=ax)
ax.set(title='Samples of Prior Distribution')
ax.legend(loc='lower right');

#--------------------------------------------
d= 4
y = X[:,2:6]

def compute_gpr_parameters(K, K_star2, K_star, sigma_n):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), y.reshape([n, d])))
    # Covariance.
    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), K_star.T))
    
    return (f_bar_star, cov_f_star)

# Compute posterior mean and covariance. 
f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, sigma_n)

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=cov_f_star, cmap='Blues', ax=ax)
ax.set_title('Components of the Covariance Matrix cov_f_star');

#--------------------------------------------

fig, ax = plt.subplots()

for i in range(0, 100):
    # Sample from posterior distribution. 
    z_star = np.random.multivariate_normal(mean=f_bar_star[:,0].squeeze(), cov=cov_f_star)
    # Plot function.
    sns.lineplot(x=x_star[:,0], y=z_star, color="blue", alpha=0.2, ax=ax);
    
# Plot "true" linear fit.
sns.scatterplot(x=x[:,0], y=X[:,2], color='red', label = 'f(x)', ax=ax)
ax.set(title=f'Samples of Posterior Distribution, sigma_f = {sigma_f} and l = {l}')
ax.legend(loc='upper right');

#--------------------------------------------
matplotlib.rcParams['figure.figsize'] = (10,10)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
plt.pcolormesh(x_star[:,0].reshape(10,10), x_star[:,1].reshape(10,10), z_star.reshape(10,10), transform = ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))
plt.scatter(data['lon'], data['lat'], c = data['chl'], s = 150, marker = '+', transform = ccrs.PlateCarree())

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

# Add Colorbar
plt.clim(-1,1)
cbar = plt.colorbar(fraction=0.03, pad=0.04)
cbar.set_label('mg', fontsize = 18)

# Add Title
plt.title('Concentration Chlorophyll-a', fontsize = 18)
plt.show()

a=2; b=0.2; sigma=np.sqrt(2); d=1
def cov(h, tau):
    cov = sigma**2 * np.exp(-b**2 * np.linalg.norm(h)**2/(a**2 * tau**2 + 1)) / (a**2 * tau**2 + 1)**(d/2)
    return cov


COV = [cov(i, j) for (i, j) in itertools.product([2,6], [0.2,1.0])]
